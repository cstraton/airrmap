# REST API service.

# %% Imports
# from flask import Flask
# from flask import make_response
# from flask_cors import CORS
from quart import Quart
from quart import make_response
from quart_cors import cors
from quart import Response
from quart import request
from quart import send_from_directory

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cProfile
import sqlite3
import gzip
import json
import time
import yaml
import io
import os

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from typing import Any, Dict, List
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import anchors
import airrmap.preprocessing.indexer as indexer
import airrmap.application.tile
import airrmap.application.converters as converters
import airrmap.application.config as config
import airrmap.application.reporting as reporting
import airrmap.preprocessing.compute_selected_coords as compute_coords
from airrmap.application.tile import TileHelper
from airrmap.application.repo import DataRepo
from airrmap.shared import models
from airrmap.application.kde import KDERelativeMode, KDERelativeSelection, KDEItem, KDEGroup, KDEManager
from airrmap.shared.timing import Timing

# %% Module level vars
# app = Flask(__name__)
app = Quart(__name__, static_url_path='', static_folder=r'static')
# CORS(app)  # Allow any origin, any route. (for cross-origin errors in browser).
cors(app)
class_rgb = None
v_group_le = None
scaler_xy = None
data_repo: DataRepo
tile_helper: TileHelper
last_num_bins = 0
tile_mask: Image.Image
num_classes = 0
request_ctr = 0
kde_manager = KDEManager()
cfg = config.AppConfig()
env_name = cfg.default_environment


def _static_folder():
    """
    Get static folder path
    """
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static')


def _compute_seq_coords(env_name: str, seq_list_text: str, cfg: config.AppConfig, scaler_xy) -> List[Dict]:
    """
    Compute the coordinates for a small list of sequences.

    Parameters
    ----------
    env_name : str
        The name of the environment.

    seq_list_text : str
        The text list of sequences, one sequence per row. The format should match
        the format used for the anchor sequences in the environment. Can be
        JSON format (the first item will be tested).

    cfg : config.AppConfig
        The application configuration to use, primarily to
        get the base path of the environment data.

    scaler_xy : [type]
        The transform to convert from the original coordinates to plot space.

    Returns
    -------
    List[Dict]
        List of dictionaries with the following keys:
            seq: The original sequence passed.
            x: The x coordinate, in plot space.
            y: The y coordinate, in plot space.

    Raises
    ------
    ValueError
        If no sequences are provided.
    """

    # Check if provided
    seq_list_text = seq_list_text.strip()
    if len(seq_list_text) == 0:
        raise ValueError('No sequences provided')

    # Split to lines
    seq_list = seq_list_text.splitlines(keepends=False)

    # Check if JSON format (checks first line)
    try:
        json.loads(seq_list[0])
        convert_json = True
    except Exception as ex:
        convert_json = False

    # Compute the coordinates
    result_list = compute_coords.get_coords(
        env_name=env_name,
        seq_list=seq_list,
        convert_json=convert_json,
        app_cfg=cfg
    )

    # Convert the sequences to plot space
    plot_x = []
    plot_y = []
    for result in result_list:
        x = result['sys_coords_x']
        y = result['sys_coords_y']
        x_tf, y_tf = scaler_xy.transform([[x], [y]])
        plot_x.append(x_tf[0])  # unwrap from array
        plot_y.append(y_tf[0])

    # Return result
    final_list = []
    for seq, x, y in zip(seq_list, plot_x, plot_y):
        rec = dict(
            seq=seq,
            x=x,
            y=y,
        )
        final_list.append(rec)
    return final_list


def _build_tile_binned(tile_request: models.TileRequest,
                       df_facet: pd.DataFrame,
                       cfg: config.AppConfig,
                       num_bins: int,
                       num_classes: int,
                       class_rgb,
                       t: Timing = None) -> Image.Image:
    """
    Compute the 2D histogram for a tile and return the
    rendered image.
    """

    # Init timer
    if t is None:
        t = Timing()

    # Get the binned data
    abinned = TileHelper.get_tile_binned(
        zoom=tile_request.tile_zoom,
        x=tile_request.tile_x,
        y=tile_request.tile_y,
        tile_size=tile_request.tile_size,
        num_bins=tile_request.num_bins,
        df_world=df_facet,
        world_x=cfg.column_plot_x,
        world_y=cfg.column_plot_y,
        world_value=cfg.column_value,
        statistic=cfg.default_statistic,
        world_class=cfg.column_class,
        num_classes=int(num_classes)
    )
    t.add('Binned tile values computed.')

    # Get the image
    im_binned = TileHelper.get_tile_image(
        binned_data=abinned,
        value1_min=tile_request.value1_min,
        value1_max=tile_request.value1_max,
        class_rgb=class_rgb,
        image_size=tile_request.tile_size,
        tile_mask=None,
        brightness=tile_request.brightness
    )
    t.add('Binned tile image created.')

    # Draw tile debug information
    imdraw = ImageDraw.Draw(im_binned)
    if cfg.tile_debug:
        # imfont = ImageFont.truetype('sans-serif.ttf', 16)
        imdraw.text(
            (10, 10),
            f'zoom:{tile_request.tile_zoom} x:{str(tile_request.tile_x)} y:{str(tile_request.tile_y)}',
            (255, 255, 255)
        )
        t.add('Binned tile debug values rendered.')

    # Return
    return im_binned


# %% Routes
@app.route('/')
async def home():
    """Home/application page (index.html)"""
    static_folder = _static_folder()
    return await send_from_directory(static_folder, 'index.html')


@app.route('/sys/hello/')
async def hello():
    """Test endpoint"""
    return "Welcome to AIRR Map."


@app.route('/examples/items/')
async def get_example_items():
    """Testing only, return a list of items"""
    items = []
    items.append(dict(id=1, name='item1'))
    items.append(dict(id=2, name='item2'))
    result = dict(items=items)

    return Response(
        json.dumps(result),
        status=200,
        mimetype='application/json'
    )


@app.route('/anchors/<env_name>/', methods=['GET'])
async def get_anchor_list(env_name):
    """Return the list of anchors for the given environment."""
    global cfg
    global scaler_xy
    return anchors.get_anchors_json(fn_anchors=cfg.get_anchordb(env_name),
                                    scaler_xy=scaler_xy,
                                    v_group_le=v_group_le,
                                    class_rgb=class_rgb)


@app.route('/lookup/kde/options/', methods=['GET'])
async def lookup_kde_options():
    """
    Get the KDE options for the front end dropdown lists.

    Returns
    -------
    Dict
        kde_relative_mode: List of options for KDERelativeMode.
        kde_relative_selection: List of options for KDERelativeSelection.
        kde_colormaps: List of valid colormap names.
    """

    # Init
    result = {}

    # Build result (enums)
    for i in range(2):
        enum_def = KDERelativeMode if i == 0 else KDERelativeSelection
        key_name = 'kde_relative_mode' if i == 0 else 'kde_relative_selection'

        item_list = []
        for el in enum_def:
            item = dict(
                key=el.name,
                text=el.name,
                value=el.name,
            )
            item_list.append(item)
        result[key_name] = item_list

    # Colormaps
    cm_list = []
    cms_sorted = sorted(plt.colormaps(), key=str.lower)
    for cm_name in cms_sorted:

        # Skip reverse colour maps
        # (we have a separate invert param)
        if cm_name[-2:] == '_r':
            continue

        # Init item and add
        item = dict(
            key=cm_name,
            text=cm_name,
            value=cm_name
        )
        cm_list.append(item)
    result['kde_colormap'] = cm_list

    # Return
    result_json = json.dumps(result)
    return result_json


@app.route('/lookup/env/', methods=['GET'])
async def lookup_env_list():
    """Get available environment names for the dropdown list."""
    cfg = config.AppConfig()
    env_data: dict = indexer.get_env_list(cfg.index_db)

    # Build list in format required for dropdowns
    env_list = []
    for x in env_data:
        item = dict(
            key=x['name'],
            text=x['name'],
            value=x['name'],
        )
        env_list.append(item)

    # JSON encode
    env_list_json = json.dumps(env_list)

    # Return
    return Response(
        env_list_json,
        status=200,
        mimetype='application/json'
    )


@app.route('/lookup/env/<env_name>/fields/', methods=['GET'])
async def lookup_field_list(env_name):
    """Get the list of available fields"""
    cfg = config.AppConfig()

    #  Default
    if env_name == 'abc':
        env_name = cfg.default_environment

     # Build list in format required for dropdowns
    field_data = indexer.get_filters(cfg.index_db, env_name)
    field_list = []
    for x in field_data:
        item = dict(
            key=x['value'],
            text=x['value'],
            value=x['value']
        )
        field_list.append(item)

    # JSON encode
    field_list_json = json.dumps(field_list).encode('utf-8')

    # Return
    return Response(
        field_list_json,
        status=200,
        mimetype='application/json'
    )


@ app.route('/index/', methods=['GET'])
async def index_read():
    """Get the index"""
    cfg = config.AppConfig()
    index_data: dict = indexer.get_index(cfg.index_db)
    index_data = json.dumps(index_data)

    return Response(
        index_data,
        status=200,
        mimetype='application/json'
    )


@app.route('/index/env/', methods=['GET'])
async def index_env_list():
    """Get list of available environment names"""
    cfg = config.AppConfig()
    env_data: dict = indexer.get_env_list(cfg.index_db)
    env_data = json.dumps(env_data)

    return Response(
        env_data,
        status=200,
        mimetype='application/json'
    )


@app.route('/index/env/<env_name>/filters/<filter_name>/', methods=['GET'])
async def index_filter_values(env_name, filter_name):
    """Get details of a filter including the type and valid values."""
    cfg = config.AppConfig()
    result = indexer.get_filter_item(cfg.index_db, env_name, filter_name)
    result_json = json.dumps(result)
    return Response(
        result_json,
        status=200,
        mimetype='application/json'
    )


@app.route('/index/filters/', methods=['GET'])
@app.route('/index/env/<env_name>/filters/', methods=['GET'])
async def index_filters(env_name=''):
    """Get the list of fields/properties that can be filtered on."""
    cfg = config.AppConfig()
    filters = indexer.get_filters(cfg.index_db, env_name)
    filters_json = json.dumps(filters)
    return Response(
        filters_json,
        status=200,
        mimetype='application/json'
    )


@app.route('/items/seqcoords/', methods=['POST'])
async def get_seq_coords():
    """
    Compute 2D coordinates for the list of sequences.
    Sequences should in the same format as the other sequences
    used in the data pipeline stage for the environment.
    """

    # Init
    global scaler_xy
    global cfg

    # Check POST request
    if not request.method == 'POST':
        return 'Only POST accepted'

    # Should be dict containing env_name and seqs text property.
    # {env_name='my_env', seqs:'AY...\nAR'}
    request_data = await request.get_json()
    env_name = request_data['env_name']
    seqs = request_data['seqs']

    # Compute coordinates
    async def async_generator():
        result: List[Dict] = _compute_seq_coords(
            env_name=env_name,
            seq_list_text=seqs,
            cfg=cfg,
            scaler_xy=scaler_xy
        )

        # Compress
        result_compressed = gzip.compress(
            json.dumps(result, default=vars).encode('utf-8'),
            7
        )

        yield result_compressed

    # Return
    return async_generator(), 200, \
        {
            'content-type': 'application/json',
            'content-encoding': 'gzip'
    }


@app.route('/items/polyroi/', methods=['POST'])
async def get_polyroi_summary():
    """
    Request a summary report for the selected polygon region.

    Gets the sys_point_ids for the current selection, then
    loads additional record properties from file.

    Parameters (post)
    -----------------
    latlngs: Dict[List[Dict[str, int...]]]
        Array of dicts, with map latitude/longtidue (x/y) points
        defining the polygon selection,
        e.g. [{'lat':1, 'lng':2}, {'lat':3...0}]

    facetRowValue: str
        Facet row value for the selected map, or '' if not used.

    facetColValue: str
        Facet col value for the selected map, or '' if not used.

    bounds:
        The rectangular bounds of the selection.

    filters : Dict
        The environment and filter selections, defining the source
        of data to be used.
        e.g.
        ```
        {
            'env_name': 'MY_ENV_NAME',
            'filters': [
                {'filter_name': 'f.Chain', 'filter_value': 'Heavy' ],
                {'filter_name': 'f.BSource', 'filter_value': 'PBMC'],
                ['filter_name': 'r.v', 'filter_value': 'IGHV3-7*02']
            ]
        }
        ```

    Yields
    -------
    logo: str
        The base64 encoded sequence logo image (png).

    seqs: List[str]
        List of sequences.

    """

    # Init
    global scaler_xy
    global data_repo
    global cfg

    #  Check POST request
    if not request.method == 'POST':
        return 'Only POST accepted'

    # Should be array of dicts
    # [{'lat':1, 'lng:2}, {'lat':...}]
    request_data = await request.get_json()

    # Get env_name and filters
    # NOTE: query.to_dict() should be the same dictionary used
    #       to run the main data query, in order to
    #       utilise the DataRepo cache.
    query = models.RepoQuery(**request_data['filters'])
    env_name = query.env_name
    facet_row_value = request_data['facetRowValue']
    facet_col_value = request_data['facetColValue']
    env_config = cfg.get_env_config(env_name)
    cdr3_field = env_config['application']['cdr3_field']
    redundancy_field = env_config['application']['redundancy_field']
    numbered_seq_field = env_config['application']['numbered_seq_field']
    v_field = env_config['application']['v_field']
    d_field = env_config['application']['d_field']
    j_field = env_config['application']['j_field']

    # Define regions to use for the sequence logos
    # TODO: Make configurable.
    # NOTE: Region should not contain chain letter ('cdr1' not 'cdrh1').
    # regions = env_config['anchor']['regions']
    regions = ['cdr1', 'cdr2', 'cdr3', 'cdr1-2']

    async def async_generator():

        #
        # --- Region coordinates ---
        #

        # Start timer
        t = Timing()

        # Convert to x, y
        # [[2, 1], [...]]
        latlngs = request_data['latlngs']
        xy_list = converters.latlng_to_xy(latlngs)

        # Last element needs to match first (closed polygon)
        if xy_list[-1] != xy_list[0]:
            xy_list.append(xy_list[0])

        # Convert from map to original coordinates
        # (also converts to NumPy array)
        xy_list = scaler_xy.inverse_transform(xy_list)

        # Get rectangular bounds of selection
        # bounds format: {'_southWest': {'lat': 21.211, 'lng':23.21}, '_northEast': {'lat': ...}}
        bounds = request_data['bounds']
        bounds_southWest = bounds['_southWest']
        bounds_northEast = bounds['_northEast']

        # Convert from map to original coordinates
        # Wrap and unwrap 2D array required for MinMaxScaler
        bounds_bottom = scaler_xy.inverse_transform(
            [[bounds_southWest['lat']]])[0][0]
        bounds_left = scaler_xy.inverse_transform(
            [[bounds_southWest['lng']]])[0][0]
        bounds_top = scaler_xy.inverse_transform(
            [[bounds_northEast['lat']]])[0][0]
        bounds_right = scaler_xy.inverse_transform(
            [[bounds_northEast['lng']]])[0][0]
        t.add('Region coordinates finished.')

        # Build field list to load
        # Filter out 'None'
        field_list = [
            cdr3_field,
            redundancy_field,
            # numbered_seq_field, # Temporarily removed, slow to load (large amount of data)
            v_field,
            d_field,
            j_field,
            'seq_gapped_cdr1',  # TODO: Make configurable
            'seq_gapped_cdr2',
            'seq_gapped_cdr3'
        ]
        field_list = [x for x in field_list if x is not None]

        # Load the report data
        df_records = data_repo.load_report_data(
            query=query,
            cfg=cfg,
            bounds_bottom=bounds_bottom,
            bounds_left=bounds_left,
            bounds_top=bounds_top,
            bounds_right=bounds_right,
            xy_lasso=xy_list,
            fields=field_list,
            t=t,
            facet_row_value=facet_row_value,
            facet_col_value=facet_col_value
        )
        t.add('Report data loaded.')

        # Add on cdr1-2 concatenated gapped sequence
        # TODO: Remove if no longer required
        df_records['seq_gapped_cdr1-2'] = df_records.apply(
            lambda row: row['seq_gapped_cdr1'] + row['seq_gapped_cdr2'],
            axis='columns'
        )
        t.add('Added cdr1-2 concatenated field.')

        # Create the report
        report = reporting.get_summary_report(
            df_records=df_records,
            regions=regions,
            facet_row_value=facet_row_value,
            facet_col_value=facet_col_value,
            cdr3_field=cdr3_field,
            redundancy_field=redundancy_field,
            v_field=v_field,
            d_field=d_field,
            j_field=j_field,
            seq_list_sample_size=50,
            t=t
        )
        t.add('Report created.')

        # Add timing information
        report['_debug'] = {}
        report['_debug']['timing'] = t

        # Compress
        report_compressed = gzip.compress(
            json.dumps(report, default=vars).encode('utf-8'),
            7
        )

        # Return
        yield report_compressed

    # Return it
    return async_generator(), 200, \
        {
            'content-type': 'application/json',
            'content-encoding': 'gzip'
    }


@app.route('/tiles/data', methods=['GET'])
async def query_data():
    """
    Called by the UI when user submits a query request.

    The resulting dataset will be cached by the query engine
    to be used for future render and report requests.
    A query report will be returned to the UI
    containing values for the rows/columns of the
    facet grid (alongside other information).
    """

    # Check for query argument
    if 'q' in request.args:
        query_raw = request.args['q']
        query = json.loads(query_raw)

    else:
        raise Exception('No query (q) argument passed')

    # Run the query
    async def async_generator():
        global cfg
        global class_rgb
        global data_repo
        global scaler_xy
        global v_group_le
        global num_classes

        # Init
        t = Timing()

        # Run the query and get the report
        df, report = data_repo.run_coords_query(
            cfg=cfg,
            query=query,
            scaler_xy=scaler_xy,
            value2_le=v_group_le,
            t=t)

        # Return the report (but not the dataset, leave cached)
        report_json = json.dumps(report).encode('utf-8')
        
        # Show timing information
        if cfg.tile_debug:
            print(json.dumps(t, default=vars, indent=2, sort_keys=True))

        yield report_json

    return async_generator(), \
        200, \
        {'content-type': 'application/json'}


@app.route('/tiles/imagery/<tile_type>/<zoom>/<x>/<y>.png', methods=['GET'])
async def get_tile_image(tile_type: str, zoom: int = 0, x: int = 0, y: int = 0):
    """
    Render a map tile image (KDE or 2D histogram).

    This is the main function called by the Leaflet map components.
    It dynamically renders the PNG images for the KDE and 2D histogram layers.
    """

    # Init
    global cfg
    global last_num_bins
    global tile_mask
    global data_repo
    global tile_helper
    global request_ctr
    global class_rgb
    global scaler_xy
    global v_group_le
    global num_classes
    query: models.RepoQuery  # = None
    request_ctr += 1
    t = Timing()
    t.add('get_tile_image() requested.', request.args)
    # print('Request ' + str(request_ctr))
    # print (str(request.args))

    # Tile request type
    tile_type = tile_type.upper().strip()
    if tile_type not in ('KDE-DIFF', 'KDE-SIMILAR', 'BINNED'):
        raise ValueError(f'Invalid tile_type: {tile_type}')

    # Brightness multiplier
    if 'brightness' in request.args:
        brightness = max(float(request.args['brightness']), 1.0)
    else:
        brightness = 1.0

    # KDE brightness multiplier
    if 'kdebrightness' in request.args:
        kdebrightness = max(float(request.args['kdebrightness']), 1.0)
    else:
        kdebrightness = 1.0

    # KDE bandwidth
    if 'kdebw' in request.args:
        kdebw = max(float(request.args['kdebw']), 0.01)
    else:
        kdebw = 1.0

    # Query / filters
    # This should be the same query that was originally
    # submitted by the user, to utilise the query engine cache
    # (otherwise the query engine will perform a new query).
    if 'q' in request.args:
        query_raw = request.args['q']
        query_d = json.loads(query_raw)

        # Return placeholder tile if
        # environment is not selected
        # (e.g. first time application is loaded)
        if 'env_name' not in query_d:
            with open(os.path.join(os.path.dirname(__file__), 'tile_placeholder.png'), 'rb') as f:
                imgbuf = io.BytesIO(f.read())
            return imgbuf, 200, {'content-type': 'image/png'}

        # Place query into RepoQuery instance
        query = models.RepoQuery(**query_d)
    else:
        raise ValueError(f"Query argument 'q' not supplied.")

    # KDE colour map
    if 'kdecm' in request.args:
        kde_colormap = request.args['kdecm']
    else:
        kde_colormap = 'RdBu'

    # KDE invert colour map
    if 'kdecminv' in request.args:
        kde_colormap_invert = True if request.args['kdecminv'].lower(
        ) == 'true' else False
    else:
        kde_colormap_invert = False

    # KDE relative mode
    # Default to SINGLE (self, no relative values)
    if 'kderm' in request.args:
        kde_rel_mode = KDERelativeMode[request.args['kderm']]
    else:
        # kde_rel_mode = KDERelativeMode.SINGLE
        #  TODO: Change back, or add NONE to KDERelativeSelection (1-1 = 0)
        kde_rel_mode = KDERelativeMode.ALL

    # KDE Selection mode
    # Default to ALL (self, no relative values)
    if 'kdesm' in request.args:
        kde_rel_selection = KDERelativeSelection[request.args['kdesm']]
    else:
        kde_rel_selection = KDERelativeSelection.ALL

    # KDE relative Row name
    # Default to '' (self, no relative value)
    if 'kderowname' in request.args:
        kde_rel_row_name = request.args['kderowname']
    else:
        kde_rel_row_name = KDEGroup.NO_SELECTION

    # KDE relative Column name
    # Default to '' (self, no relative value)
    if 'kdecolname' in request.args:
        kde_rel_col_name = request.args['kdecolname']
    else:
        kde_rel_col_name = KDEGroup.NO_SELECTION

    # Get facet (r)ow/(c)ol values
    facet_row_value = request.args['fr'] if 'fr' in request.args else ''
    facet_col_value = request.args['fc'] if 'fc' in request.args else ''

    # Get value1 min/max
    # (for scaling brightness)
    value1_min = 0
    value1_max = 88  # TODO: Config?
    if 'v1min' in request.args:
        value1_min = request.args['v1min']
    if 'v1max' in request.args:
        value1_max = request.args['v1max']

    # Number of bins
    # Will be provided as 0, 1 etc.
    # 0 = 256, 1 = 128, 2 = 64 etc.
    if 'bins' in request.args:
        num_bins = int(request.args['bins'])
        num_bins = 256 / (2**num_bins)
    else:
        num_bins = cfg.default_num_bins

    # Round to integer (required for tile binning)
    num_bins = int(round(num_bins))

    # TODO: Tidy / move
    # Leaflet CRS.Simple workaround:
    # Translate tile Y coordinate
    # for use with Leaflet CRS.Simple
    # coordinate system.
    # x=0,y=0 becomes x=0,y=-1, with -ve moving up.
    # So take max squares high (2^zoom)
    # and subtract abs(y)
    y = 2**int(zoom) - abs(int(y))

    # Get circles mask (show 2D bins as circles instead of squares).
    # (currently not used, slows rendering performance.
    if num_bins != last_num_bins:
        # tile_mask = tile.TileHelper.get_circle_mask(ts.tile_size, num_bins)
        tile_mask = None

    t.add('Initialisation finished.')

    # Run the query and get the sequence data.
    # (this data will typically be pre-cached by the query engine due to the
    # original user request).
    # allow_cached_report = True: we don't mind if the query times
    # in the report are not correct; we're just
    # rendering image tiles and not showing the times anywhere.
    df, report, d_query_data_split = data_repo.run_coords_query(
        cfg=cfg,
        query=query.to_dict(),
        scaler_xy=scaler_xy,
        value2_le=v_group_le,
        split_by_facet=True,
        allow_cached_report=True,
        t = None
    )
    t.add('Dataset loaded.')

    # Get query hash from the report
    query_hash = report['query_hash']

    # Wrap everything in a tile request
    tile_request: models.TileRequest = models.TileRequest(
        tile_type=tile_type,
        tile_x=x,
        tile_y=y,
        tile_zoom=zoom,
        tile_size=int(cfg.default_tile_size),
        query=query.to_dict(),
        query_hash=query_hash,
        brightness=brightness,
        facet_row_value=facet_row_value,
        facet_col_value=facet_col_value,
        value1_min=value1_min,
        value1_max=value1_max,
        value2_min=0.,  # not used
        value2_max=0.,  # not used
        num_bins=num_bins,
        kdebrightness=kdebrightness,
        kdebw=kdebw,
        kde_colormap=kde_colormap,
        kde_colormap_invert=kde_colormap_invert,
        kde_rel_mode=kde_rel_mode,
        kde_rel_selection=kde_rel_selection,
        kde_rel_row_name=kde_rel_row_name,
        kde_rel_col_name=kde_rel_col_name
    )
    t.add('Tile request created.')

    async def async_generator():

        # Filter on facet subset if required
        # ('' for all values / not used).
        df_facet = d_query_data_split[(
            tile_request.facet_row_value,
            tile_request.facet_col_value
        )]
        t.add('Facet data selected.')

        # Default to False
        invert_values = False

        # Get kde image
        if tile_request.tile_type == 'KDE-DIFF' or tile_request.tile_type == 'KDE-SIMILAR':

            # Compute or get the cached KDETileSet with KDEs for ALL facets
            # (required for computation of relative values)
            kde_tileset, tileset_cached = kde_manager.get_tileset(
                tile_request=tile_request,
                d_world_split=d_query_data_split,
                facet_row_values=report['facet_row_values'],
                facet_col_values=report['facet_col_values'],
                world_x=cfg.column_plot_x,
                world_y=cfg.column_plot_y,
                world_value=cfg.column_value
            )
            t.add('KDE Tileset loaded from KDE Manager.',
                  {'tileset_cached': tileset_cached})

            # Get the KDEItem from the KDETileSet
            kde_items_d = kde_tileset.kde_items
            kde_item = kde_items_d[(facet_row_value, facet_col_value)]
            t.add('KDE Item extracted from Tileset.')

            # Get the min-max ranges
            if not kde_tileset.kde_diff_similar_minmax_computed:
                raise Exception(
                    'Expecting KDETileSet.kde_diff_minmax_computed to be True')
            diff_min, diff_max, similar_min, similar_max = kde_tileset.kde_diff_similar_minmax
            if tile_request.tile_type == 'KDE-DIFF':
                kde_grid = kde_item.kde_diff
                source_value_min, source_value_max = (diff_min, diff_max)
            elif tile_request.tile_type == 'KDE-SIMILAR':
                kde_grid = kde_item.kde_similar
                source_value_min, source_value_max = (similar_min, similar_max)
            else:
                raise Exception(
                    f'Unexpected tile type: {tile_request.tile_type}')
            t.add('Min-max range obtained.')

            # Produce an image from the kde matrix values
            # TODO: Cache min/max?
            im_kde = tile_helper.array_to_image(
                grid=kde_grid,
                source_value_min=source_value_min,
                source_value_max=source_value_max,
                brightness_multiplier=tile_request.kdebrightness,
                range_min=0.0,  # Image pixel min
                range_max=1.0,  # Image pixel max
                invert_values=tile_request.kde_colormap_invert,
                cmap=tile_request.kde_colormap
            )
            t.add('KDE image created.')

            # Convert to RGBA
            # (previously needed alpha channel for compositing)
            im = im_kde.convert('RGBA')
            t.add('KDE converted to RGBA.')

        # Get 2D histogram / binned image
        elif tile_request.tile_type == 'BINNED':

            # Get binned image
            im = _build_tile_binned(
                tile_request=tile_request,
                df_facet=df_facet,
                cfg=cfg,
                num_bins=tile_request.num_bins,
                num_classes=num_classes,
                class_rgb=class_rgb,
                t=t
            )
        else:
            raise ValueError(f'Unexpected tile_type: {tile_type}')

        # Write the PNG image data
        # im.save('my_image.png', format='png')
        with io.BytesIO() as output:
            im.save(output, format='PNG')
            t.add('Tile image written (BytesIO).')
            if cfg.tile_debug:
                print(json.dumps(t, default=vars, indent=2, sort_keys=True))
            yield output.getvalue()

    return async_generator(), 200, {'content-type': 'image/png'}


# %%


def init_test_card(*args):
    """Coloured striped test card to test rendering of tiles"""

    # Generate test data
    df, num_classes, class_rgb = tile.TileHelper.get_test_card()

    # Load into tile set
    ts = tile.TileSet(df, num_classes=num_classes, class_column='value2')

    # Return
    return ts, class_rgb, df


# %%
def get_xy_scaler() -> MinMaxScaler:
    """Scale dataframe with x and y columns in-place"""

    # Note: Will clip
    # TODO: Check clip, and dynamically adjust feature range to data

    # Range -must- match tile size, which must match world size
    # (but not necessarily number of bins).
    target_range = (0, 256)
    scaler = MinMaxScaler(feature_range=target_range, copy=False, clip=True)

    data_range = cfg.coordinates_range
    scaler.fit(np.array(data_range)[:, np.newaxis])  # reshape to 2D for fit()
    return scaler


# %% Main
# if __name__ == '__main__':


# Init
print("Initialising...")
scaler_xy = get_xy_scaler()
data_repo = DataRepo(cfg)
tile_helper = TileHelper()
# TODO: make dynamic from categorical values
v_groups = [f'IGHV{i+1}' for i in range(8)]
# v_groups.extend([f'IGLV{i+1}' for i in range(8)])
v_group_le = LabelEncoder().fit(v_groups)

# Get colours for label encodings (list of RGB tuples)
print("Getting colours...")
num_classes = len(v_group_le.classes_)
v_group_rgb = sns.mpl_palette("Set2", num_classes, as_cmap=False)
class_rgb = np.array(v_group_rgb)
# ts, class_rgb, df = init_test_card()
print("Initialised")

# Start the server
from werkzeug.middleware.profiler import ProfilerMiddleware
# app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[5], profile_dir='./profile')

if __name__ == '__main__':
    # app.run(debug=True)
    # app.run(debug=False)  # , host='0.0.0.0')
    app.run(debug=False, host='0.0.0.0')
