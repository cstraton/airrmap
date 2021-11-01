# Query Engine.
# Functionality to handle reading and transformation of data from file.
# All queries should utilise this module.

# tests -> test_repo.py (wip)

# %% imports
import sqlite3
import pandas as pd
import numpy as np
import sqlescapy
import os
import hashlib
import math
import json
import time
from collections import OrderedDict
from typing import Any, Dict, List, OrderedDict, Sequence, Tuple
from tqdm import tqdm

import airrmap.application.polypoint as polypoint
from airrmap.application import config as config
from airrmap.shared.timing import Timing
from airrmap.shared.models import RepoQuery
from airrmap.shared.cache import Cache

# %% Define class
class DataRepo:

    # Elements for query cache tuple
    QUERY_CACHE_FULL_DF = 0
    QUERY_CACHE_SPLIT_FACET = 1

    def __init__(self, cfg: config.AppConfig):
        """Construct a new DataRepo instance.

        Args:
            cfg (AppConfig): Initialised AppConfig instance.
        """
        self.cfg: config.AppConfig = cfg
        self._query_cache: OrderedDict = OrderedDict()
        self.query_cache_size = 1 # TODO: Make configurable, max number of query results to store in memory
        self._report_cache: Cache = Cache(self.query_cache_size)

    @staticmethod
    def _dict_hash(dictionary: Dict[str, Any]) -> str:
        """MD5 hash of a dictionary."""
        # Taken from: https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html

        dhash = hashlib.md5()
        # We need to sort arguments so {'a': 1, 'b': 2} is
        # the same as {'b': 2, 'a': 1}
        encoded = json.dumps(dictionary, sort_keys=True).encode()
        dhash.update(encoded)
        return dhash.hexdigest()

    # --- Query cache ---

    def clear_query_cache(self):
        """Clear the query cache"""
        self._query_cache.clear()
        self._report_cache.clear()

    # ------------------

    @staticmethod
    def select_rect(df: pd.DataFrame,
                    x_column: str,
                    y_column: str,
                    top: float,
                    right: float,
                    bottom: float,
                    left: float,
                    top_inclusive=True,
                    right_inclusive=True,
                    bottom_inclusive=True,
                    left_inclusive=True) -> pd.DataFrame:
        """
        Selects a rectangular region of a DataFrame.

        NOTE: Assumes higher y values are towards the top.

        Parameters
        ----------
        df : pd.DataFrame
            The existing DataFrame.

        x_column : str
            Column containing the x coordinates.

        y_column : str
            Column containing the y coordinates.
            Higher y values should be towards the top.

        top : float
            The top boundary of the rectangle.

        right : float
            The right boundary of the rectangle.

        bottom : float
            The bottom boundary of the rectangle.

        left : float
            The left boundary of the rectangle.

        top_inclusive : bool, optional
            True if the top boundary should be inclusive,
            otherwise False. By default, True.

        right_inclusive : bool, optional
            True if the right boundary should be inclusive,
            otherwise False. By default, True.

        bottom_inclusive : bool, optional
            True if the bottom boundary should be inclusive,
            otherwise False. By default, True.

        left_inclusive : bool, optional
            True if the left boundary should be inclusive,
            otherwise False. By default, True.

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame, containing only records that
            fall within the requested region.
        """

        # Handle inclusive/exclusive (Pandas inclusive only allows both or none)
        top = top if top_inclusive == True else np.nextafter(
            top, -math.inf)  # below slightly
        right = right if right_inclusive == True else np.nextafter(
            right, -math.inf)  # left slightly
        bottom = bottom if bottom_inclusive == True else np.nextafter(
            bottom, math.inf)  # above slightly
        left = left if left_inclusive == True else np.nextafter(
            left, math.inf)  # right slightly

        # Get the region
        df_roi = df[
            df[x_column].between(left, right, inclusive=True) &
            df[y_column].between(bottom, top, inclusive=True)
        ]

        return df_roi

    @staticmethod
    def coords_roi(df_coords: pd.DataFrame, polygon) -> pd.DataFrame:
        """
        Return points inside the given area.

        Parameters
        ----------
        df_coords : DataFrame
            DataFrame containing the list of coordinates. Coordinates
            should match the coordinate reference system (CRS) used
            by polygon.

        polygon : NumPy array
            Array of [x,y] polygon points defining the region.
            Attention! Last point in array should match the first
            (required for polypoint.is_inside_sm_parallel())

        Returns
        -------
        pd.DataFrame
            Points falling within the area or on the boundary.
        """

        if not np.array_equal(polygon[-1], polygon[0]):
            raise Exception('Last element should match first')

        # True/False filter
        in_roi = polypoint.is_inside_sm_parallel(
            polygon=polygon,
            points=df_coords[['x', 'y']].values
        )

        # Return
        return df_coords[in_roi]

    def select_records_by_id(self,
                             env_name: str,
                             fields: Sequence[str],
                             point_ids: Sequence) -> pd.DataFrame:
        """
        Load records for given list of sequence/point IDs.

        Parameters
        ----------
        env_name : str
            Environment name.

        fields : Sequence[str]
            Fields that should be retrieved.

        point_ids : Sequence
            The list of point IDs to filter for.

        Returns
        -------
        pd.DataFrame
            The filtered dataset.

        Raises
        ------
        Exception
            If not all point IDs could be found.
        """

        # Init
        fdr = self.cfg.get_seqdb_parquet_folder(env_name=env_name)
        filters = [('sys_point_id', 'in', point_ids)]

        # Read in the data, matching on point id
        df = pd.read_parquet(path=fdr,
                             columns=fields,
                             filters=filters)

        # Check all point IDs are accounted for
        if len(df.index) < len(point_ids):
            raise Exception(
                'Some point ids could not be found in the record file(s).')

        return df

    def load_report_data(self,
                         query: RepoQuery,
                         cfg: config.AppConfig,
                         bounds_bottom: float,
                         bounds_left: float,
                         bounds_top: float,
                         bounds_right: float,
                         xy_lasso: List[List[float]],
                         fields: List[str] = [],
                         t: Timing = Timing(),
                         facet_row_value: str='',
                         facet_col_value: str='') -> pd.DataFrame:
        """
        Load the data to be used in region-of-interest reports.

        Reports may require additional fields that are not loaded into the 
        'coords' DataFrame (which is kept slim to improve speed/reduce memory consumption). 
        Additionally, this method filters records to a selected region.

        Process
        -------
        1. Uses cached 'coords' data from the current query to get a list of
            sys_point_id values currently in scope (defined by user filters). Loads the
            data if it isn't cached.

        2. Further reduces the scope to points within the rectangle/lasso selected regions.

        3. Loads records from disk with requested fields, filtered by sys_point_id values.

        4. Adds on the facet_row and facet_col columns from the 'coords' data.

        Parameters
        ----------
        query : RepoQuery
            The existing query filter that the user has applied.
            The filtered records (coordinates) will typically be
            cached in data_repo, and are used to define the scope
            of the report prior to region filtering.

        cfg : AppConfig
            The application configuration instance.

        bounds_bottom : float
            Bottom of rect bounds for the selected region, in original coordinates.

        bounds_left : float
            Left of rect bounds for the selected region, in original coordinates.

        bounds_top : float
            Top of rect bounds for the selected region, in original coordinates.

        bounds_right : float
            Right of rect bounds for the selected region, in original coordinates.

        xy_lasso : List[List[float]]
            List of x,y coordinates defining the polygon points of the lasso selection,
            in original coordinates.

        fields : List[str], optional
            List of record fields to load, required by the report. By default [].

        t : Timing, optional
            Timing instance for profiling, by default Timing().

        facet_row_value : str, optional
            The row value of the facet that was selected.
            By default NOT_SELECTED (all facets).

        facet_col_value : str, optional
            The column value of the facet that was selected.
            By default NOT_SELECTED (all facets).

        Returns
        -------
        pd.DataFrame
            Records filtered for the selected region, with the requested fields.
        """

        NOT_SELECTED = ''

        #
        # --- Data filtering ---
        #

        t.add('Start get_report_data().')
        data_repo: DataRepo = self

        # Get the coordinates
        # Will usually be cached, as report usually requested after a query run.
        df_coords, report, d_query_data_split = data_repo.run_coords_query(
            cfg=cfg, 
            query=query.to_dict(),
            split_by_facet=True,
            allow_cached_report=True,
            t = None)
        t.add('Run coords query.')

        # If facet provided, then just get the data for that facet.
        if facet_row_value != NOT_SELECTED and \
            facet_col_value != NOT_SELECTED:
            facet_key_tuple = (facet_row_value, facet_col_value)
            #df_coords = df_coords[(df_coords['facet_row'] == facet_row_value) & (df_coords['facet_col']==facet_col_value)].copy()
            df_coords = d_query_data_split[facet_key_tuple]

        # Filter to records in rectangular region
        df_coords = data_repo.select_rect(
            df=df_coords,
            x_column='x',
            y_column='y',
            top=bounds_top,
            right=bounds_right,
            bottom=bounds_bottom,
            left=bounds_left,
            top_inclusive=True,
            right_inclusive=True,
            bottom_inclusive=True,
            left_inclusive=True
        )
        t.add('Filter to rectangular region.')

        # Lasso-selection to get points inside region
        # (runs after faster rectangle filter)
        df_coords = data_repo.coords_roi(df_coords, xy_lasso)
        t.add('Filter to lasso region.')

        # Convert to list of point ids
        point_ids = list(df_coords['sys_point_id'])
        t.add('Get list of point ids.')

        #
        # --- Load files ---
        #

        # Get all records in the region
        # Gets additional fields not in df_coords
        # Select by id to ensure filters by data_repo are
        # taken into account.
        fields.insert(0, 'sys_point_id')
        df_records = data_repo.select_records_by_id(
            env_name=query.env_name,
            fields=fields,
            point_ids=point_ids
        )  # ,
        # limit=n) # limit for dev
        t.add('Load records by point id.')

        # Set indexes for join
        df_coords.set_index('sys_point_id', inplace=True)
        df_records.set_index('sys_point_id', inplace=True)

        # Add on facet_row and facet_col from the original DataFrame
        df_records = df_records.join(df_coords[['facet_row', 'facet_col']])

        # Reset indexes
        df_records.reset_index(inplace=True)
        t.add('Add facet row/col fields.')
        t.add('Finished get_report_data().')

        # Return
        return df_records

    @staticmethod
    def split_by_facet(df: pd.DataFrame,
                       facet_row_column='facet_row',
                       facet_col_column='facet_col') -> Dict[Tuple[Any, Any], pd.DataFrame]:
        """
        Split a dataframe by the facet values.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to split.

        facet_row_column : str
            The column containing the facet row value.

        facet_col_column : str
            The column containing the facet column value.

        Returns
        -------
        Dict[Tuple[Any, Any], pd.DataFrame]
            Dictionary where key = (facet_row_value, facet_col_value) and
            value = DataFrame containing only records for the given
            facet row and column.
        """

        # Apply group by
        df_grouped = df.groupby(by=[facet_row_column, facet_col_column])

        # Place in dictionary
        # key = Tuple('row_value', 'col_value')
        # value = DataFrame subset for facet
        df_dict = {group_name: group for group_name, group in df_grouped}

        # Return
        return df_dict

    @staticmethod
    def filter_by_facet(df: pd.DataFrame,
                        facet_row_value='',
                        facet_col_value='') -> pd.DataFrame:
        """
        Convenience function to filter records for
        the selected facet row and/or column value.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame produced by run_coords_query().

        facet_row_value : str, optional
            Selected facet row value, by default '' (not selected).
            Case-sensitive.

        facet_col_value : str, optional
            Selected facet column value, by default '' (not selected).
            Case-sensitive.

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame.
        """

        NOT_SELECTED = ''

        if facet_row_value != NOT_SELECTED and facet_col_value != NOT_SELECTED:
            df_sub = df[(df.facet_row == facet_row_value) &
                        (df.facet_col == facet_col_value)]
        elif facet_row_value != NOT_SELECTED:
            df_sub = df[(df.facet_row == facet_row_value)]
        elif facet_col_value != NOT_SELECTED:
            df_sub = df[(df.facet_col == facet_col_value)]
        else:
            df_sub = df

        return df_sub

    @staticmethod
    def get_coords_query_report(
            df: pd.DataFrame,
            d_split_by_facet: Dict[Tuple[Any, Any], pd.DataFrame],
            success: bool,
            query: Dict,
            query_hash: str,
            cached: bool,
            start_time: float,
            end_time: float,
            value1_field: str,
            value2_field: str,
            status_message='',
            is_facet_row=False,
            facet_row_property_level='',
            facet_row_property_name='',
            facet_row_max_allowed=0,
            facet_row_sort_values=None,
            is_facet_col=False,
            facet_col_property_level='',
            facet_col_property_name='',
            facet_col_max_allowed=0,
            facet_col_sort_values=None):
        """
        Build a query report to return alongside the query results.

        Designed to show query information to the client
        and support application functionality such as
        the facet grid row/column values and the 
        min-max range for 2D histogram rendering.


        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the query result.

        d_split_by_facet : Dict[Tuple[Any, Any], pd.DataFrame],
            Same data as df, but split by facet.
            Key will be a tuple of (facet_row_value, facet_col_value).

        success : bool
            True if query ran successfully.

        query : dict
            The original query arguments.

        query_hash : str
            Hash that uniquely identifies the query. Used to
            determine if results are already cached.

        cached : bool
            True if query result was already cached, otherwise False.

        start_time : float
            CPU start time of the query.

        end_time : float
            CPU end time of the query.

        value1_field : str
            The field used for value 1.

        value2_field : str
            The field used for value 2.

        status_message : str, optional
            General message to be returned to client, by default ''.

        is_facet_row : bool, optional
            True if facet row was requested, by default False.

        facet_row_property_level: str, optional
            Facet row property level, 'f' (file) or 'r' (record). By default ''.

        facet_row_property_name: str, optional
            Facet row property name, e.g. 'Longitudinal', 'Subject'. By default ''.

        facet_row_max_allowed: int, optional
            Maximum number of values allowed for facet rows. Empty
            list of values will be returned if exceeded. Default 0.

        facet_row_sort_values: List, optional
            List of values for the facet row field, in required
            sort order, for example ['subject-4', 'subject-13']

        is_facet_col : bool, optional
            True if facet col was requested, by default False.

        facet_col_property_level: str, optional
            Facet column property level, 'f' (file) or 'r' (record). By default ''.

        facet_col_property_name: str, optional
            Facet column property name, e.g. 'Longitudinal', 'Subject'. By default ''.

        facet_col_max_allowed: int, optional
            Maximum number of values allowed for facet cols. Empty
            list of values will be returned if exceeded. Default 0.

        facet_col_sort_values: List, optional
            List of values for the facet column field, in required
            sort order, for example ['hour-1', 'day-1']


        Returns
        -------
        Dict
            The populated report.
        """

        # Init
        TIME_DIGITS = 8
        FACET_KEY_DELIM = '||'
        facet_row_sort_values = [] if facet_row_sort_values is None else facet_row_sort_values
        facet_col_sort_values = [] if facet_col_sort_values is None else facet_col_sort_values

        # Lambda functions for sorted() key param (custom sort if supplied in envconfig.yaml)
        def facet_row_sort(x): return facet_row_sort_values.index(
            x) if x in facet_row_sort_values else -1

        def facet_col_sort(x): return facet_col_sort_values.index(
            x) if x in facet_col_sort_values else -1

        # Construct the report
        # NOTE: Python maintains the order - keep this in a logical order for readability
        report: Dict[str, Any] = {}
        report['success'] = success
        report['status_message'] = status_message
        report['query'] = query
        report['query_hash'] = query_hash
        report['query_time'] = round(end_time - start_time, TIME_DIGITS)
        report['cached'] = cached
        report['record_count'] = len(df.index)
        report['value1_field'] = value1_field
        report['value1_max'] = float(df['value1'].max())
        report['value1_min'] = float(df['value1'].min())
        report['value2_field'] = value2_field

        # Facet delim (e.g. 'row||col')
        report['facet_key_delim'] = FACET_KEY_DELIM

        # Facet stats, value1 (min, max etc.)
        report['value1_facet_stats'] = {}
        for (facet_row_value, facet_col_value), df_facet in d_split_by_facet.items():
            facet_key = f'{str(facet_row_value)}{FACET_KEY_DELIM}{str(facet_col_value)}'
            facet_stats = df_facet['value1'].astype(
                'float').describe().to_dict()
            facet_stats['sum'] = float(df_facet['value1'].sum())  # Add sum
            report['value1_facet_stats'][facet_key] = facet_stats

        # Facet row
        report['facet_row_enabled'] = is_facet_row
        report['facet_row_property_level'] = facet_row_property_level
        report['facet_row_property_name'] = facet_row_property_name
        if is_facet_row:
            report['facet_row_count'] = len(df['facet_row'].cat.categories)
            report['facet_row_max_allowed'] = facet_row_max_allowed
            report['facet_row_sorted'] = len(facet_row_sort_values) > 0
            report['facet_row_values'] = sorted(list(df['facet_row'].cat.categories), key=facet_row_sort) \
                if report['facet_row_count'] <= facet_row_max_allowed else []

            if report['facet_row_count'] > facet_row_max_allowed:
                report['success'] = False
                report['status_message'] = ' '.join([
                    report['status_message'],
                    'Maximum allowed facet row values exceeded. Refine search or choose another property.'
                ])

        else:
            report['facet_row_count'] = 0
            report['facet_row_max_allowed'] = facet_row_max_allowed
            report['facet_row_sorted'] = False
            report['facet_row_values'] = []

        # Facet col
        report['facet_col_enabled'] = is_facet_col
        report['facet_col_property_level'] = facet_col_property_level
        report['facet_col_property_name'] = facet_col_property_name
        if is_facet_col:
            report['facet_col_count'] = len(df['facet_col'].cat.categories)
            report['facet_col_max_allowed'] = facet_col_max_allowed
            report['facet_col_sorted'] = len(facet_col_sort_values) > 0
            report['facet_col_values'] = sorted(list(df['facet_col'].cat.categories), key=facet_col_sort) \
                if report['facet_col_count'] <= facet_col_max_allowed else []

            if report['facet_col_count'] > facet_col_max_allowed:
                report['success'] = False
                report['status_message'] = ' '.join([
                    report['status_message'],
                    'Maximum allowed facet col values exceeded. Refine search or choose another property.'
                ])
        else:
            report['facet_col_count'] = 0
            report['facet_col_max_allowed'] = facet_col_max_allowed
            report['facet_col_sorted'] = False
            report['facet_col_values'] = []

        return report

    def run_coords_query(
            self,
            cfg: config.AppConfig,
            query: Dict,
            scaler_xy=None,
            value2_le=None,
            split_by_facet=False,
            allow_cached_report=False,
            t: Timing = None,
            ) -> Tuple:
        """
        Run a user query in the given environment.

        This is the main function used to pull data for the client,
        and returns a filtered set of sequence coordinates suitable for tile
        rendering. Queries are restricted to the same environment,
        so that distance measures and positioning are consistent.

        Two steps:
        Step 1 takes the list of file-level query filters (name and value)
        and finds all in-scope sequence files, via a lookup to the
        Metadata Index db for the specified environment. This step also builds
        the filters necessary for the record level filtering.

        Step 2 reads the list of files using Parquet and record/row
        level filtering. Results are cached according to the number of
        items defined in self.query_cache_size, with oldest entries
        overwritten if exceeded.

        If facet_row/facet_col fields are provided in the query, then
        filtering above is applied first.


        Parameters
        ----------
        cfg : config.AppConfig
            The initialised Config object, which provides the paths
            to the relevant source files.

        query : Dict
            A dictionary containing the environment name and filters
            (see example).

        scaler_xy : MinMaxScaler (Optional)
            Scaler to map xy coordinates into required range (e.g. 0-256).

        value2_le : LabelEncoder (Optional)
            Encoder that translates value2 text categories to integers (used for colour palette).

        split_by_facet : Bool (Optional)
            If True, the returned result will include an additional dictionary where the keys are tuples of
            (facet_row_value, facet_col_value) and the values are DataFrames
            containing the data for each facet.

        allow_cached_report : Bool (Optional)
            If True and a report has previously been generated for the
            given query, the cached report will be returned. However,
            the timings and potentially other properties may reflect
            the original query and not the cached query. False will
            ensure the report is regenerated. Typically, use True
            for tile requests and False when user submits a new query.
            By default, False.

        t : Timing (Optional)
            Timing information will be added if supplied.
            By default, False.
            
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the full results from the query.

        Dict
            A query Report including 
            distinct facet_row / facet_col values
            (required for facet rendering) and query statistics
            such as time and number of rows.

        Dict[Tuple[Any, Any], pd.DataFrame] - only if split_by_facet = True.
            A dictionary of DataFrames with the data for each facet.
            Key=(facet_row_value, facet_col_value).


        Example
        -------
        ```
        cfg = Config()
        query = {
            'env_name': 'MY_ENV_NAME',
            'value1_field': 'redundancy',  # Pass '' or exclude if not used, will use default in envconfig.yaml
            'value2_field': 'v', # Pass '' or exclude if not used, will use default in envconfig.yaml
            'facet_col': 'f.Longitudinal',  # Pass '' or exclude if not used
            'facet_row': 'f.Subject',       # Pass '' or exclude if not used
            'filters': [
                {'filter_name': 'f.Chain', 'filter_value': 'Heavy' },   # f. = file level
                {'filter_name': 'f.BSource', 'filter_value': 'PBMC'},   # f. = file level 
                {'filter_name': 'r.v', 'filter_value': 'IGHV3-7*02'}    # r. = record level
            ]
        }
        run_query(cfg, query)
        ```
        """

        # Init
        if t is None:
            t = Timing()
        t0 = time.process_time()
        RECORD_FILTER_LEVEL = 'r'
        FILE_FILTER_LEVEL = 'f'
        FILTER_NAME = 'filter_name'
        FILTER_VALUE = 'filter_value'
        NOT_SPECIFIED = ''
        t.add("Start of query process.")

        # Get the selected filters and env config
        env_name = query['env_name']
        envconfig = cfg.get_env_config(env_name)

        # Get default value1 and value2 fields from envconfig.yaml
        try:
            value1_field_default = envconfig['application']['value1_field']
        except KeyError:
            value1_field_default = None

        try:
            value2_field_default = envconfig['application']['value2_field']
        except KeyError:
            value2_field_default = None

        # Use value1 and value 2 fields if passed, otherwise try to use defaults in envconfig.yaml
        is_value1_field = 'value1_field' in query and query['value1_field'] != NOT_SPECIFIED
        is_value2_field = 'value2_field' in query and query['value2_field'] != NOT_SPECIFIED
        value1_field = query['value1_field'] if is_value1_field else value1_field_default
        value2_field = query['value2_field'] if is_value2_field else value2_field_default
        if value1_field is None:
            raise ValueError(
                'value1_field not passed in query and no default in envconfig.yaml')
        elif value2_field is None:
            raise ValueError(
                'value2_field not passed in query and no default in envconfig.yaml')
 
        # Facet row/col (if provided)
        is_facet_row = 'facet_row' in query and query['facet_row'] != NOT_SPECIFIED
        is_facet_col = 'facet_col' in query and query['facet_col'] != NOT_SPECIFIED
        facet_row = query['facet_row'] if is_facet_row else NOT_SPECIFIED
        facet_col = query['facet_col'] if is_facet_col else NOT_SPECIFIED

        # Split facet arg, e.g. 'f.field1' --> ('f', 'field1')
        if is_facet_row:
            facet_row_property_level, facet_row_property_name = \
                facet_row.split('.', 1)
            facet_row_property_level = facet_row_property_level.lower().strip()
        else:
            facet_row_property_level = NOT_SPECIFIED
            facet_row_property_name = NOT_SPECIFIED

        if is_facet_col:
            facet_col_property_level, facet_col_property_name = \
                facet_col.split('.', 1)
            facet_col_property_level = facet_col_property_level.lower().strip()
        else:
            facet_col_property_level = NOT_SPECIFIED
            facet_col_property_name = NOT_SPECIFIED

        # Check if custom sort values provided in config
        try:
            facet_row_sort_values = envconfig['plot']['sort_values'][facet_row]
        except KeyError:
            facet_row_sort_values = []

        try:
            facet_col_sort_values = envconfig['plot']['sort_values'][facet_col]
        except KeyError:
            facet_col_sort_values = []

        t.add("Query arguments retrieved.")


        # Return cached results if already exists
        # NOTE!: If updating, remember to update report at
        #       bottom of function (non-cached df return)
        query_hash = DataRepo._dict_hash(query)
        if (query_hash in self._query_cache):
            df = self._query_cache[query_hash][DataRepo.QUERY_CACHE_FULL_DF]
            d_split_by_facet = self._query_cache[query_hash][DataRepo.QUERY_CACHE_SPLIT_FACET]
            report_cached = self._report_cache.get(query_hash, if_not_exists=None)
            t1 = time.process_time()

            if report_cached is not None and allow_cached_report:
                report = report_cached
                t.add("Query report generated.", {"cached": True})
            else:
                report = DataRepo.get_coords_query_report(
                    df=df,
                    d_split_by_facet=d_split_by_facet,
                    success=True,
                    query=query,
                    query_hash=query_hash,
                    cached=True,
                    start_time=t0,
                    end_time=t1,
                    value1_field=value1_field,
                    value2_field=value2_field,
                    status_message='',
                    is_facet_row=is_facet_row,
                    facet_row_property_level=facet_row_property_level,
                    facet_row_property_name=facet_row_property_name,
                    facet_row_max_allowed=cfg.facet_row_max_allowed,
                    facet_row_sort_values=facet_row_sort_values,
                    is_facet_col=is_facet_col,
                    facet_col_property_level=facet_col_property_level,
                    facet_col_property_name=facet_col_property_name,
                    facet_col_max_allowed=cfg.facet_col_max_allowed,
                    facet_col_sort_values=facet_col_sort_values
                )
                t.add("Query report generated.", {"cached": False})

            t.add("Query completed.", {"cached": True})
            return (df, report, d_split_by_facet) if split_by_facet else (df, report)

        if 'filters' in query:
            filters = query['filters']
        else:
            filters = []

        # Hold the record-level Parquet filters
        record_parq_filters: List = []

        # File paths
        fn_indexdb = cfg.index_db
        fdr_records = cfg.get_seq_folder(env_name, config.SeqFileType.RECORD)
        fdr_records = sqlescapy.sqlescape(fdr_records)
        fdr_meta = cfg.get_seq_folder(env_name, config.SeqFileType.META)
        fdr_meta = sqlescapy.sqlescape(fdr_meta)
        fdr_env = cfg.get_env_folder(env_name)
        fdr_env = sqlescapy.sqlescape(fdr_env)

        # ***********************************************************************
        # ** STEP 1: Obtain all filter properties (file and record levels).
        #            Get the seq db files which are in scope and build
        #            record filters for the query.
        # ***********************************************************************

        # List of properties to filter by (all if none specified)
        # [Property Level].[Property Name] = property_level_name.
        # Comma separate and wrap in single quotes for SQL
        if len(filters) > 0:
            # filter_prop_names = [x[FILTER_NAME] for x in filters]
            # filter_prop_placeholders = ','.join(
            #    ['?' for x in filter_prop_names])
            # filter_props_sql = f' AND property_level_name IN ({filter_prop_placeholders})'
            filter_props_sql: str = ''

            # Filter the index on selected filters to get in-scope file names.
            # If 'file' level filter, filter on property name -and- value.
            # If 'record' level filter, filter just on property name for file
            #   and also build up the filters for the query.
            file_sql_filters: List = []
            for filter_item in filters:

                # Skip any empty or incomplete filters
                # (e.g. if UI form submitted without finishing)
                if filter_item is None or \
                    (FILTER_NAME not in filter_item) or \
                        (FILTER_VALUE not in filter_item):
                    continue

                # Split combined property_level and property_name
                # (e.g. 'f.sys_filename -> ['f', 'sys_filename'])
                property_level, property_name = filter_item[FILTER_NAME].split(
                    '.', 1)
                property_level = property_level.lower().strip()

                # index table is long - need to use INTERSECT
                # otherwise multiple filters will result in no records
                # (sys_property_name is per record, not column)
                if property_level == FILE_FILTER_LEVEL:

                    # SQL filtering for file level filtering (uses index.db)

                    # Leave as is (will be passed to SQL engine)
                    filter_op = ''
                    filter_sql_op = ''
                    filter_value = filter_item[FILTER_VALUE].strip()
                    if filter_value[:1] == '!':
                        filter_sql_op = 'NOT LIKE'
                        filter_value = filter_value[1:]
                    else:
                        filter_sql_op = 'LIKE'

                    # file level
                    filter_props_sql += f"""\n
                        INTERSECT
                        SELECT DISTINCT file_name
                        FROM env_index
                        WHERE (property_level = ?
                            AND property_name = ?
                            AND property_value {filter_sql_op} ?
                        )
                    """
                    file_sql_filters.append(property_level)
                    file_sql_filters.append(property_name)
                    file_sql_filters.append(filter_value)

                elif property_level == RECORD_FILTER_LEVEL:

                    # Parquet filters for record level.
                    # record level (just make sure file contains this field, don't check values as too many)

                    # TODO:  Actually we would prefer to error if
                    # file doesn't contain the field (it should do)
                    # filter_props_sql += """\n
                    #    INTERSECT
                    #    SELECT DISTINCT file_name
                    #    FROM env_index
                    #    WHERE (property_level = ?
                    #        AND property_name = ?)
                    # """
                    # file_sql_filters.append(property_level)
                    # file_sql_filters.append(property_name)

                    # Parquet filtering at Record level, see:
                    # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html

                    # Get filter value
                    filter_value = filter_item[FILTER_VALUE].strip()
                    if filter_value[:1] in ['=', '<', '>']:
                        filter_op = filter_value[0]
                        filter_value = filter_value[1:]
                    elif filter_value[:2] in ['==', '<=', '>=', '!=']:
                        filter_op = filter_value[:2]
                        filter_value = filter_value[2:]
                    elif filter_value[:3] == 'in(':
                        filter_op = 'in'
                        # -2; Ignore final ')'
                        filter_value = filter_value[3:-2]
                        filter_value = filter_value.split(',')
                    elif filter_value[:7] == 'not in(':
                        filter_op = 'not in'
                        # -2; Ignore final ')'
                        filter_value = filter_value[7:-2]
                        filter_value = filter_value.split(',')
                    else:
                        # Default to = if no operator
                        filter_op = '='
                        filter_value = filter_value

                    # Parquet filtering
                    # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html
                    record_parq_filters.append(
                        (property_name, filter_op, filter_value))
                else:
                    raise Exception(
                        'Unexpected property level: ' + str(property_level))

        else:
            # filter_prop_names = []
            file_sql_filters = []
            filter_props_sql = ''

        # Get the list of seq db files that meet the
        # env name and list of filter property names
        # from the index db.
        sql_file_list = f"""
            SELECT file_name
            FROM   env_index
            WHERE  env_name = ?
                 {filter_props_sql}
            GROUP BY file_name
            ORDER BY file_name
        """

        sql_params = [env_name]
        # sql_params.extend(filter_prop_names)
        sql_params.extend(file_sql_filters)
        t.add("Query preparation completed.")

        # Get the -record- filenames that fall in-scope for the given filter.
        # Don't create db file if doesn't exist
        with sqlite3.connect(f'file:{fn_indexdb}?mode=ro', uri=True) as conn:
            curr = conn.cursor()
            files = [file_name
                     for file_name
                     in curr.execute(sql_file_list, sql_params).fetchall()]
            t.add("In-scope file list loaded from Metadata Index.")

            # Check if any files match the given filters
            if files == []:
                raise Exception('No files match the given criteria.')
                return

            file_list = list(zip(*files))[0]  # List of tuples to one tuple

        # ****************************************************
        # *** STEP 2: Load record data from the in-scope files
        # ****************************************************

        # Get columns to read
        columns = [
            'sys_point_id',
            'sys_coords_x',
            'sys_coords_y',
            value1_field,
            value2_field
        ]

        # If facet row/column is file-level
        # we'll need sys_file_id to pull in file meta
        if facet_row_property_level == FILE_FILTER_LEVEL or \
                facet_col_property_level == FILE_FILTER_LEVEL:
            columns.append('sys_file_id')

        # Also read in requested facet
        # row/column fields if record-level
        facet_row_exclusive = False
        facet_col_exclusive = False
        if is_facet_row and facet_row_property_level == RECORD_FILTER_LEVEL:
            if facet_row_property_name not in columns:
                columns.append(facet_row_property_name)
                facet_row_exclusive = True
        if is_facet_col and facet_col_property_level == RECORD_FILTER_LEVEL:
            if facet_col_property_name not in columns:
                columns.append(facet_col_property_name)
                facet_col_exclusive = True

        # Get list of Parquet files (seq records and file meta)
        # If only 1 record, remove from array (get single element as str)
        # otherwise Parquet will throw error in pd.read_parquet()
        # For meta files, should end with .meta.parquet.
        fns_records: Any = [os.path.join(
            fdr_records, fn) for fn in file_list]
        fns_records = fns_records if len(fns_records) > 1 else fns_records[0]
        fns_meta: Any = [os.path.join(
            fdr_meta, fn.replace('.parquet', '.meta.parquet')) for fn in file_list]
        fns_meta = fns_meta if len(fns_meta) > 1 else fns_meta[0]
        t.add("Parquet file and column details retrieved.")

        # Read -record- Parquet files
        df = pd.read_parquet(
            path=fns_records,
            columns=columns,
            filters=record_parq_filters if len(
                record_parq_filters) > 0 else None,  # filtering
            memory_map=False,
            buffer_size=0
        )
        t.add("Parquet file data loaded to DataFrame.")

        # Add on the facet fields to the main
        # records DataFrame (as category type).
        # If -record- level, then simply add the new column.
        # as 'facet_row' / 'facet_col'.
        # If -file- level, then read in the
        # Parquet meta files (for each Data Unit),
        # filtering on the selected property.
        # Then merge / join on sys_file_id and add
        # to the main records DataFrame.

        # Set up the list of facet values to add
        # (will loop through these next...)
        facet_items = []

        facet_items.append(
            (facet_row_property_level,
                facet_row_property_name,
                'facet_row',
                facet_row_exclusive)
        )

        facet_items.append(
            (facet_col_property_level,
                facet_col_property_name,
                'facet_col',
                facet_col_exclusive)
        )

        # Loop through and add facet values to df
        for facet_item in facet_items:

            facet_property_level = facet_item[0]
            facet_property_name = facet_item[1]
            facet_df_column = facet_item[2]
            facet_field_exclusive = facet_item[3]

            # Add facet_row / facet_col as category column
            # (should be limited number)
            if facet_property_level == RECORD_FILTER_LEVEL:
                df[facet_df_column] = df[facet_property_name].astype(
                    'category')

                # If we only read in a
                # field specifically for the
                # facet row / column, we can
                # delete the original column now...
                if facet_field_exclusive:
                    df.drop(
                        [facet_property_name],
                        axis=1,
                        inplace=True
                    )

            elif facet_property_level == FILE_FILTER_LEVEL:
                # Read -meta- Parquet files
                df_meta: pd.DataFrame = pd.read_parquet(
                    path=fns_meta,
                    columns=[
                        'sys_file_id',
                        'property_name',  # Filter below only
                        'property_value'
                    ],
                    filters=[('property_name', '=', facet_property_name)]
                )

                # Join by sys_file_id
                # and add on the file-level
                # property values.
                df_meta.drop_duplicates(inplace=True)
                df_meta['property_value'] = df_meta['property_value'].astype(
                    'category')
                df_meta.rename(
                    columns={'property_value': facet_df_column},
                    inplace=True
                )

                # Join on sys_file_id, pull back facet_row/facet_column
                meta_cols = ['sys_file_id', facet_df_column]
                df = df.merge(df_meta[meta_cols], how='left', copy=False)

            elif facet_property_level == NOT_SPECIFIED:
                # Add empty field if not supplied
                # (cleaner for groupby shortly...)
                df[facet_df_column] = NOT_SPECIFIED
                df[facet_df_column] = df[facet_df_column].astype('category')
        t.add("Facet values added to DataFrame.")

        # If facet row/column is file-level;
        # can remove sys_file_id we temporarily added
        if facet_row_property_level == FILE_FILTER_LEVEL or \
                facet_col_property_level == FILE_FILTER_LEVEL:
            df.drop(
                ['sys_file_id'],
                axis=1,
                inplace=True
            )

        # Rename columns
        col_mapping = {
            'sys_coords_x': 'x',
            'sys_coords_y': 'y',
            value1_field: 'value1',
            value2_field: 'value2'
        }
        df.rename(columns=col_mapping, inplace=True)
        t.add("Tidying columns completed.")

        # -------------------------------------------

        # TODO: move back to x? double memory storing as x_scaled.
        #       but also y reversed (below).
        if scaler_xy is not None:
            df['x_tf'] = scaler_xy.transform(df[['x']].values)
            df['y_tf'] = scaler_xy.transform(df[['y']].values)
            t.add("Added scaled X/Y columns.", {"computed": True})
        else:
            df['x_tf'] = df['x']
            df['y_tf'] = df['y']
            t.add("Added scaled X/Y columns.", {"computed": False})

        # reverse y, so higher y values are down.
        # must be performed - after- scaling.
        #df['y_tf'] = df['y_tf'].max() - df['y_tf']
        df['y_tf'] = 256.0 - df['y_tf']  # Should be maximum world y allowed
        t.add("Reversed scaled Y column")

        # Negate y - in tile world space, higher y is down,
        # but Leaflet map L.CRS coordinate system, higher y is up.
        #df['y_tf'] = -df['y_tf']

        # Reduce memory usage of columns
        # before caching
        column_size_convert = dict(
            x='float32',
            y='float32',
            value1='float32',
            value2='category',
            x_tf='float32',
            y_tf='float32',
        )
        df = df.astype(column_size_convert)
        t.add("Changed column data types (reduce memory usage).")

        # TODO: Temporay, remove
        # %% Label encode category value.
        if value2_le is not None:
            df['class_index'] = value2_le.transform(
                df['value2'].apply(lambda x: x[:5]))  # First 5, e.g. IGHV3))
            df['class_index'] = df['class_index'].astype('uint32')
            t.add("Added class_index (value2 label encoder).", {"computed": True})
        else:
            df['class_index'] = df['value2']
            t.add("Added class_index (value2 label encoder).", {"computed": False})

        # Create DataFrame split by facet
        # Always compute and store in the cache
        # regardless of split_by_facet option, so
        # we will have it in the cache in case
        # future requests require it.
        # Adds about 500-700ms in testing.
        d_split_by_facet = DataRepo.split_by_facet(
            df=df,
            facet_row_column='facet_row',
            facet_col_column='facet_col'
        )
        t.add("Facet split completed.")

        # Store in the cache (full df and split by facet)
        self._query_cache[query_hash] = (df, d_split_by_facet)

        # Remove oldest/first item if over the cache size
        if len(self._query_cache) > self.query_cache_size:
            oldest_item = self._query_cache.popitem(last=False)
        t.add("Cache operations completed.")

        # Build report
        # NOTE: If updating, remember to update report at
        #       top of function (cached df return)
        t1 = time.process_time()
        report = DataRepo.get_coords_query_report(
            df=df,
            d_split_by_facet=d_split_by_facet,
            success=True,
            query=query,
            query_hash=query_hash,
            cached=False,
            start_time=t0,
            end_time=t1,
            value1_field=value1_field,
            value2_field=value2_field,
            status_message='',
            is_facet_row=is_facet_row,
            facet_row_property_level=facet_row_property_level,
            facet_row_property_name=facet_row_property_name,
            facet_row_max_allowed=cfg.facet_row_max_allowed,
            facet_row_sort_values=facet_row_sort_values,
            is_facet_col=is_facet_col,
            facet_col_property_level=facet_col_property_level,
            facet_col_property_name=facet_col_property_name,
            facet_col_max_allowed=cfg.facet_col_max_allowed,
            facet_col_sort_values=facet_col_sort_values
        )
        t.add("Query report generated.")


        # Store report in the cache
        self._report_cache.store(query_hash, report)

        # Return
        t.add("Query completed.", {"cached": False})
        return (df, report, d_split_by_facet) if split_by_facet else (df, report)