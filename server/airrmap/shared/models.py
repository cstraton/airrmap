"""
Model classes / structures
"""

from typing import Any, Dict, List
from airrmap.application.kde_enum import KDERelativeMode, KDERelativeSelection

class ReportItem():

    def __init__(self,
                name: str,
                title: str,
                report_type: str,
                data: Any,
                x_label: str='',
                y_label: str=''):
        """
        A single report item.

        Returned as part of a parent report.
        Typically json.dumps(default=vars) will be called,
        returning all properties of this class.

        Parameters
        ----------
        name : str
            Name of this specific instance, unique within
            the parent report.

        title : str
            Title (e.g. for plot heading).

        report_type : str
            Type of report.

        data : List[Dict] or Dict.
            Report data. Items should be
            in DataFrame.to_dict(orient='list') format,
            suitable for loading into Plotly.js.
            If grouped by facet, each item should have 
            a 'facet_row_value' and 'facet_col_value' property.

        
        x_label : str (optional)
            Label for x axis, by default ''.

        y_label : str (optional)
            Label for y axis, by default ''.
        """

        self.name=name
        self.title=title
        self.report_type=report_type
        self.x_label=x_label
        self.y_label=y_label
        self.data=data



class RepoQuery():

    def __init__(
            self,
            env_name: str,
            value1_field='',
            value2_field='',
            facet_col='',
            facet_row='',
            filters=None):
        """
        A query instance.

        Parameters
        ----------
        env_name : str
            Name of the environment to run the query for.

        value1_field : str, optional
            Field for value1. If not supplied, will use
            default in envconfig. By default ''.

        value2_field : str, optional
            Field for value2. If not supplied, will use
            default in envconfig. By default ''.

        facet_col : str, optional
            Field to group by for the facet column, by default ''.

        facet_row : str, optional
            Field to group by for the facet row, by default ''.

        filters : list, optional
            List of dictionaries with the following keys:

            filter_name: The property name. This can be (f)ile meta level
                (e.g. 'f.Chain') or (r)ecord level (e.g. 'r.v').

            filter_value: The value to filter for / criteria.
        """

        self.env_name = env_name
        self.value1_field = value1_field
        self.value2_field = value2_field
        self.facet_col = facet_col
        self.facet_row = facet_row
        self.filters = [] if filters is None else filters

    def to_dict(self):
        return self.__dict__


class TileRequest():

    def __init__(
        self,
        tile_type: str,
        tile_x: int,
        tile_y: int,
        tile_zoom: int,
        tile_size: int,
        query: Dict,
        query_hash: str,
        brightness: float,
        facet_row_value: str,
        facet_col_value: str,
        value1_min: float,
        value1_max: float,
        value2_min: float,
        value2_max: float,
        num_bins: int,
        color_field: str,
        kdebrightness: float,
        kdebw: float,
        kde_colormap: str,
        kde_colormap_invert: bool,
        kde_rel_mode: KDERelativeMode,
        kde_rel_selection: KDERelativeSelection,
        kde_rel_row_name: str,
        kde_rel_col_name: str):
        """
        Represents a request for a single tile.

        Parameters
        ----------
        tile_type : str
            Type of image tile to retrieve.
            Valid values are 'KDE' or 'BINNED'.

        tile_x : int
            The tile x coordinate.

        tile_y : int
            The tile y coordiante.

        tile_zoom : int
            The zoom level (0 for whole world).

        tile_size: int
            Width/height of the square tile in pixels.

        query : Dict
            Dictionary containing the query to run.
            This determines the filtering of the data.
            See RepoQuery() for more details. 
        
        query_hash : str
            Generated hash for the query dictionary
            (used for caching).

        brightness : float
            Brightness multiplier.

        facet_row_value : str
            The facet row value the request relates to.

        facet_col_value : str
            The facet column value the request relates to.

        value1_min : float
            Minimum value, value 1. 
            Used for plot brightness/colour scaling.
            Typically returned in the report provided
            by from the repo query.

        value1_max : float
            Maximum value, value 1. 
            Used for plot brightness/colour scaling.
            Typically returned in the report provided
            by from the repo query.

        value2_min : float
            Minimum value, value 2.

        value2_max : float
            Maximum value, value2.

        num_bins : int
            Number of bins requested for the tile
            (heatmap layer, horizontal and vertical bins).

        color_field : str
            The field that will determine the heatmap colors.

        kdebrightness : float
            KDE brightness multiplier.

        kdebw : float
            KDE bandwidth adjustment.       

        kde_colormap: str
            Name of the colour map to use.

        kde_colormap_invert: bool
            True to invert the colour map.

        kde_rel_mode : KDERelativeMode
            The KDE mode to use if computing relative values.
            See KDERelativeMode for options.

        kde_rel_selection : KDERelativeSelection
            The KDE selection method to use if computing
            relative values. See KDERelativeSelection for options.

        kde_rel_row_name : str
            The name of the row to use if computing relative values, or
            KDEGroup.NO_SELECTION for all rows or same as given KDE (depends on kde_rel_mode).

        kde_rel_col_name : str
            The name of the column to use if computing relative values, or
            KDEGroup.NO_SELECTION for all columns or same as given KDE (depends on kde_rel_mode).
        """

        # NOTE: 1 .If changing any KDE properties that affect KDE grid values,
        #       remember to update KDEManager.get_tileset_key()
        #       (not required if only affects rendered image, e.g. brightness).
        # 
        #       2. Remember to update get_dummy_instance() function.
        self.tile_type = tile_type.upper().strip()
        self.tile_x = int(tile_x)
        self.tile_y = int(tile_y)
        self.tile_zoom = int(tile_zoom)
        self.tile_size = int(tile_size)
        self.query = query
        self.query_hash = query_hash
        self.brightness = float(brightness)
        self.facet_row_value = facet_row_value
        self.facet_col_value = facet_col_value
        self.value1_min = float(value1_min)
        self.value1_max = float(value1_max)
        self.value2_min = float(value2_min)
        self.value2_max = float(value2_max)
        self.num_bins = int(num_bins)
        self.color_field = color_field
        self.kdebrightness = float(kdebrightness)
        self.kdebw = float(kdebw)
        self.kde_colormap = kde_colormap
        self.kde_colormap_invert = kde_colormap_invert
        self.kde_rel_mode = kde_rel_mode
        self.kde_rel_selection = kde_rel_selection
        self.kde_rel_row_name = kde_rel_row_name
        self.kde_rel_col_name = kde_rel_col_name
        
    @classmethod
    def get_dummy_instance(cls):
        """
        Empty TileRequest instance for general testing.

        This function creates a TileRequest instance with
        dummy values. Default values are kept out of the
        constructor as all values should be supplied;
        an error should occur if otherwise.

        Returns
        -------
        TileRequest
            A TileRequest instance with dummy values.
        """
        item = cls(
            tile_type='tile_type',
            tile_x=0,
            tile_y=0,
            tile_zoom=0,
            tile_size=256,
            query=dict(),
            query_hash='abc',
            brightness=0.,
            facet_row_value='',
            facet_col_value='',
            value1_min=0.,
            value1_max=1.,
            value2_min=0.,
            value2_max=1.,
            num_bins=256,
            color_field='',
            kdebrightness=1.,
            kdebw=.75,
            kde_colormap = 'RdBu',
            kde_colormap_invert = False,
            kde_rel_mode = KDERelativeMode.ALL,
            kde_rel_selection = KDERelativeSelection.FIRST,
            kde_rel_row_name='kde_rel_row_name',
            kde_rel_col_name='kde_rel_col_name'
        )
        return item

   


