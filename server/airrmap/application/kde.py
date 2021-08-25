# KDE-related functionality.

# > tests > test_kde.py

# %% Imports
import itertools
import unittest
import pandas as pd
import numpy as np
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.interpolate import RectBivariateSpline

from airrmap.shared.models import TileRequest
from airrmap.shared.cache import Cache
from airrmap.application import faststats_kde as fskde
from airrmap.application.tile import TileHelper
from airrmap.application.kde_enum import KDERelativeMode, KDERelativeSelection
from airrmap.application.repo import DataRepo


class KDEItem():

    KEY_DELIM = '|'
    KEY_PREFIX = f'k{KEY_DELIM}'

    def __init__(
            self,
            kde_abs: np.ndarray,
            facet_row_value: Union[str, None],
            facet_col_value: Union[str, None],
            kde_extents: Optional[Tuple] = None,
            kde_group=None,  # Left KDEGroup type as not yet defined.
            kde_norm: Optional[np.ndarray] = None,
            kde_diff: Optional[np.ndarray] = None,
            kde_similar: Optional[np.ndarray] = None):
        """
        Simple class to hold properties and computed KDE(s) for a given
        facet.

        Parameters
        ----------
        kde_abs : np.ndarray
            The unnormalized 2D KDE grid.

        facet_row_value : Union[str, None]
            The facet row value, or None (if grid of zeros).

        facet_col_value : Union[str, None]
            The facet column value, or None (if grid of zeros).

        kde_extents : Tuple (xmin, xmax, ymin, ymax)
            Tuple of the extents of kde_abs grid, by default None.

        kde_group: KDEGroup (optional)
            The KDE group this has been assigned to, by default None.

        kde_norm : np.ndarray (optional)
            The normalized 2D KDE grid, where the sum
            of values equals 1.0. If None, it will be computed automatically.
            By default None.

        kde_diff : np.ndarray (optional)
            The 2D KDE grid, containing differences relative to a selected
            population (e.g. a control KDE).

        kde_similar : np.ndarray (optional)
            The 2D KDE grid, containing similarities relative to a selected
            population (e.g. a control KDE).
        """

        self.key = KDEItem.get_key(facet_row_value, facet_col_value)
        self.facet_row_value = facet_row_value
        self.facet_col_value = facet_col_value
        self.kde_extents = kde_extents
        self.kde_group = kde_group
        self.kde_abs = kde_abs
        self.kde_norm = kde_norm if kde_norm is not None else KDEItem.normalize(
            kde_abs)
        self.kde_diff = kde_diff
        self.kde_similar = kde_similar

    @staticmethod
    def get_key(facet_row_value: Union[str, None], facet_col_value: Union[str, None]) -> str:
        """Generate a composite key"""
        facet_row_key = facet_row_value if facet_row_value is not None else f'__NONE'
        facet_col_key = facet_col_value if facet_col_value is not None else f'__NONE'
        return f'{KDEItem.KEY_PREFIX}{facet_row_key}{KDEItem.KEY_DELIM}{facet_col_key}'

    @staticmethod
    def normalize(x):
        """
        Normalize 2D array to sum to 1.0.

        Parameters
        ----------
        x : ndarray
            The 2D array.

        Returns
        -------
        ndarray
            The normalized array, that sums to 1.0.
            If input is 0.0, returns the original array.
        """

        if x.sum() == 0:
            return x

        return x / x.sum()

    @staticmethod
    def get_tile_fskde(zoom: int, x: int, y: int,
                       tile_size: int,
                       df_world: pd.DataFrame,
                       use_zscore: bool = False,
                       world_x: str = 'x',
                       world_y: str = 'y',
                       world_value: str = 'value1',
                       adjust_bw: float = 1.0) -> Tuple[np.ndarray, Tuple]:
        """
        Compute the kde for the given tile.

        Parameters
        ----------
        zoom : int
            Tile zoom level.
        x : int
            Tile x coordinate.
        y : int
            Tile y coordinate.
        tile_size : int
            Size of tile (e.g. 256 px) and grid size for the kde.
        df_world : pd.DataFrame
            Pre-filtered world data for the -current facet-.
        use_zscore: bool
            True to return grid values as z-scores, otherwise False. By default False.
        world_x : str, optional
            Column name of x world-space coordinates (e.g. 0-256), by default 'x'.
        world_y : str, optional
            Column name of y world-space coordinates (e.g. 0-256), by default 'y'.
        world_value : str, optional
            Column name containing the measure values, by default 'value1'.
        adjust_bw : float, optional
            Bandwidth scale adjustment, by default 1.0.

        Returns
        -------
        ndarray:
            The computed KDE grid, of shape (tile_size, tile_size).

        Tuple (xmin, xmax, ymin, ymax):
            The extents of the world-space coordinates represented
            by the KDE grid.
        """

       # Get world coordinates from tile
        (x_from, y_from), (x_to, y_to) = TileHelper.tile_to_world(
            zoom, x, y, tile_size)

        # Filter data to just this region.
        # Note inclusive=True, but right side is exclusive
        # due to math.nextafter() workaround to ensure we don't double up data points
        # exactly on the right/bottom border. Assumes world coordinates always >= 0.0)
        # REF: https://stackoverflow.com/questions/6063755/increment-a-python-floating-point-value-by-the-smallest-possible-amount

        # TODO: Optimize: don't use &
        # REF: https://jakevdp.github.io/PythonDataScienceHandbook/03.12-performance-eval-and-query.html
        df_roi = df_world[df_world[world_x].between(x_from, np.nextafter(x_to, -math.inf),
                                                    inclusive=True) &
                          df_world[world_y].between(y_from, np.nextafter(y_to, -math.inf),
                                                    inclusive=True)]

        # %% Run kde
        kde_grid, kde_extents = fskde.fastkde(
            x=df_roi[world_x].values,
            y=df_roi[world_y].values,
            nocorrelation=False,
            gridsize=(tile_size, tile_size),
            extents=(x_from, x_to, y_from, y_to),
            weights=df_roi[world_value].values,
            adjust=adjust_bw  # Affects time
        )

        # %% Compute z score, use ddof=0 for population stddev
        # Centered around 0.
        if use_zscore:
            kde_grid = (kde_grid - kde_grid.mean()) / kde_grid.std(ddof=0)

        # Return
        return kde_grid, kde_extents

    @staticmethod
    def interp_kde_region(zoom: int, x: int, y: int,
                          tile_size: int,
                          kde_world_grid):
        """
        Interpolates a portion of the kde grid for the given tile.

        This is an optimisation so that KDE only has to be computed
        once at the world level, but provides a smooth/non-pixelated
        image as the user zooms in.

        It takes a KDE grid computed for the whole 'world',
        extracts the square region associated with the selected tile,
        and uses interpolation to produce a square grid the size
        of tile_size.


        Parameters
        ----------
        zoom : int
            Tile zoom level.
        x : int
            Tile x coordinate.
        y : int
            Tile y coordinate.
        tile_size : int
            Size of the tile (e.g. 256 pixels).
        kde_world_grid : ndarray
            2D KDE grid for whole of world space
            regardless of selected tile.

        Returns
        -------
        ndarray
            2D interpolated KDE of shape (tile_size, tile_size)
        """

        # Get world coordinates from tile
        (x_from, y_from), (x_to, y_to) = TileHelper.tile_to_world(
            zoom, x, y, tile_size)

        # kde_world should represent whole world space (e.g. 0-256)
        # Get just the rectangular region we're interested in.
        # Should be 1 'pixel'/'square' on the kde_world grid
        # per coordinate in world space.
        x_coarse = np.arange(x_from, x_to, 1.)
        y_coarse = np.arange(y_from, y_to, 1.)
        Z_coarse = kde_world_grid[
            int(x_from):int(x_to),
            int(y_from):int(y_to)

        ]

        # Fit spline for interpolation
        interp_spline = RectBivariateSpline(
            x_coarse, y_coarse, Z_coarse
        )

        # Apply interpolation
        x_fine = np.linspace(
            start=x_from,
            stop=x_to,
            num=tile_size,
            endpoint=False
        )
        y_fine = np.linspace(
            start=y_from,
            stop=y_to,
            num=tile_size,
            endpoint=False
        )
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        Z_fine = interp_spline(x_fine, y_fine)

        # Return the new grid
        return Z_fine

    def compute_diff(self,
                     kde_norm: np.ndarray,
                     relative_kde: np.ndarray,
                     store_diff: bool = False) -> np.ndarray:
        """
        Compute the difference between a kde
        and another kde (compute relative values).

        kde_norm : ndarray
            A normalised kde grid array, where values sum to 1.0.

        relative_kde : ndarray
            The kde to compute the values relative to.
            Should be normalised, where values sum to 1.0.

        store_diff : bool (Optional)
            If True, will store the result in this instance's
            kde_diff property, by default False.

        Returns
        -------
        ndarray
            The difference between the kdes, in the range -1.0 to +1.0.
        """

        kde_diff = kde_norm - relative_kde

        if store_diff:
            self.kde_diff = kde_diff

        return kde_diff

    def compute_similar(self,
                        kde_norm: np.ndarray,
                        relative_kde: np.ndarray,
                        store_similar: bool = False) -> np.ndarray:
        """
        Compute the similarity between a kde
        and another kde (compute relative values).

        kde_norm : ndarray
            A normalised kde grid array, where values sum to 1.0.

        relative_kde : ndarray
            The kde to compute the values relative to.
            Should be normalised, where values sum to 1.0.

        store_similar : bool (Optional)
            If True, will store the result in this instance's
            kde_similar property, by default False.

        Returns
        -------
        ndarray
            The similarity between the kdes, in the range 0.0 to +1.0
            where +1.0 is exactly the same.
        """

        kde_similar = 1.0 - np.abs(kde_norm - relative_kde)

        if store_similar:
            self.kde_similar = kde_similar

        return kde_similar


class KDEGroup():

    NO_SELECTION = ''
    KEY_DELIM = '|'
    KEY_PREFIX = f'k{KEY_DELIM}'

    def __init__(self,
                 kde_items: List[KDEItem],
                 facet_row_value: Union[str, None] = NO_SELECTION,
                 facet_col_value: Union[str, None] = NO_SELECTION):
        """
        Represents a group of KDEs.

        Filters and stores in-scope items upon instantiation.
        Mean of KDEs can be computed.

        Parameters
        ----------
        kde_items : List[KDEItem]
            List of all KDE instances.

        facet_row_value : Union[str, None], optional
            The row of items this group represents, NO_SELECTION
            for all rows, or None to compare to grid of zeros (self), where
            size of the grid will be determined by the first item in kde_items.
            By default NO_SELECTION.

        facet_col_value : Union[str, None], optional
            The column of items this group represents, or NO_SELECTION
            for all columns, or None to compare to grid of zeros (self), where
            size of the grid will be determined by the first item in kde_items.
            By default NO_SELECTION.
        """

        # Filter the list to only ones that belong in this group,
        # or create a grid of zeros for no relative comparison.
        first_kde_item = kde_items[0]

        # If both facet_row_value and facet_col_value are None,
        # set to grid of zeros (to compare KDE to nothing).
        if facet_row_value is None and facet_col_value is None:
            kde_shape = first_kde_item.kde_abs.shape
            group_kdes: List[KDEItem] = [
                KDEItem(
                    kde_abs=np.zeros(kde_shape),
                    facet_row_value=None,
                    facet_col_value=None,
                )
            ]
        elif (facet_row_value is None and facet_col_value is not None) or \
                (facet_col_value is None and facet_row_value is not None):
            raise ValueError(
                'Both facet_row_value or facet_col_value must be None, if either is None.')
        else:
            group_kdes = KDEGroup.filter_group_kdes(
                kde_items=kde_items,
                facet_row_value=str(facet_row_value),
                facet_col_value=str(facet_col_value)
            )

        # If self
        self.is_self_group = True if facet_row_value is None and facet_col_value is None else False

        # Create key and store
        self.key = KDEGroup.get_group_key(facet_row_value, facet_col_value)
        self.group_kde_items = group_kdes
        self.facet_row_value = facet_row_value
        self.facet_col_value = facet_col_value
        self.is_group_kde_computed = False
        self.group_kde: Optional[np.ndarray] = None

    def compute_group_kde(self, force=False):
        """
        Compute a 2D KDE whose values are the medians
        of the KDEs in this group.

        KDEs in this group should have been normalised beforehand, 
        wher sum of the KDE sums to 1.0.

        Parameters
        ----------
        force : bool, optional
            This group may be assigned to multiple KDEItem instances and
            this method may be called multiple times. As the result is automatically
            cached, the cached result will be returned unless force=True.
            NOTE: No checks are currently performed to invalidate the cache
            if any of this group's properties change; is_group_kde_computed
            should be explicitly reset to False if this occurs, to ensure
            the KDE is re-computed.
            By default False.

        Returns
        -------
        ndarray
            A 2D ndarray, representing the medians of the KDEs in this group.
        """

        # Skip if already computed
        if self.is_group_kde_computed and not force:
            return

        # Get the KDE ndarrays
        kde_norm_items = tuple([x.kde_norm for x in self.group_kde_items])

        # Stack 2D KDEs into 3D Numpy array along 3rd axis (axis 2)
        kde_stacked = np.dstack(kde_norm_items)

        # Compute 2D KDE based on median of stacked items
        kde_median = np.median(kde_stacked, axis=2, keepdims=False)

        # Store and return
        self.group_kde = kde_median
        self.is_group_kde_computed = True
        return kde_median

    @staticmethod
    def get_group_key(facet_row_value: Union[str, None], facet_col_value: Union[str, None]) -> str:
        """Generate a group composite key"""
        facet_row_key = facet_row_value if facet_row_value is not None else f'__NONE'
        facet_col_key = facet_col_value if facet_col_value is not None else f'__NONE'
        return f'{KDEGroup.KEY_PREFIX}{facet_row_key}{KDEGroup.KEY_DELIM}{facet_col_key}'

    @staticmethod
    def in_group_scope(kde: KDEItem,
                       facet_row_value: str,
                       facet_col_value: str) -> bool:
        """
        Check if the given kde should be part of this group.

        Parameters
        ----------
        kde : KDEItem
            The KDE item.

        facet_row_value : str
            The row value of this group. If NO_SELECTION, all rows
            are considered in scope.

        facet_col_value : str
            The column value of this group. If NO_SELECTION, all columns
            are considered in scope.

        Returns
        -------
        bool
            True if the kde item belongs in this group, otherwise False.
        """
        if (kde.facet_row_value == facet_row_value or facet_row_value == KDEGroup.NO_SELECTION) \
                and (kde.facet_col_value == facet_col_value or facet_col_value == KDEGroup.NO_SELECTION):
            return True
        else:
            return False

    @staticmethod
    def filter_group_kdes(
            kde_items: List[KDEItem],
            facet_row_value: str,
            facet_col_value: str) -> List[KDEItem]:
        """
        Filter the list of kdes to only items that belong in this group.

        Parameters
        ----------
        kde_items : List[KDEItem]
            List of KDE Items.

        Returns
        -------
        List[KDEItem]
            The filtered list of KDE items.
        """

        # Filter on in-scope for this group
        filtered_kde_items = [x for x in kde_items
                              if KDEGroup.in_group_scope(
                                  x,
                                  facet_row_value,
                                  facet_col_value
                              )
                              ]

        # Return
        return filtered_kde_items


class KDETileSet():

    def __init__(self, key=''):
        """
        A class that represents the KDEs for all facets for
        a given tile location and query result.
        """

        self.key = key
        self.kde_items: Dict[Tuple, KDEItem] = dict()
        self.kde_diff_similar_minmax: Tuple[float, float, float, float]
        self.kde_diff_similar_minmax_computed = False

       # Compute all the absolute KDEs.
        #tileset = KDETileSet()
        # tileset.compute_kde_abs(df_world, tile_request, facet_row_values, facet_col_values) # multiprocessing
        # tileset.compute_kde_norms()  # multiprocessing.
        # tileset.assign_kde_groups() # single core
        # tileset.compute_kde_diffs() # single core
        # store it in the cache.

    def compute_kde_abs_norm(self,
                             d_world_split: Dict[Tuple, pd.DataFrame],
                             world_x: str,
                             world_y: str,
                             world_value: str,
                             tile_request: TileRequest,
                             store_items: bool = False) -> Dict[Tuple, KDEItem]:
        """
        Compute the un-normalised and normalised KDEs for all facets.

        NOTE: Will be much slower if random values are used for test data.
        Consider using fixed values with some jitter (fixed values only
        will result in a single matrix error).

        Returns
        -------
        Dict[Tuple, KDEItem]
            A dictionary containing an initialised KDEItem for each facet.
            Key = (facet_row_value, facet_col_value).
            Value = KDEItem, where the kde_abs and kde_norm property has been set.

        Parameters
        ----------
        d_world_split : Dict[Tuple, pd.DataFrame]
            A dictionary containing DataFrames for each facet.
            key=(facet_row_value, facet_col_value).

        world_x : str
            The column containing the x coordinate.

        world_y : str
            The column containing the y coordinate.

        world_value : str
            The column containing the measure value.

        tile_request : TileRequest
            The Tile Request instance.

        store_items : bool, optional
            If True, will also store the KDE Items in this KDETileSet instance,
            in addition to returning them. By default False.

        Returns
        -------
        Dict[Tuple, KDEItem]
            The computed KDEs for each facet.
            key=(facet_row_value, facet_col_value).
            value=KDEItem.
        """

        kde_items = {}
        keys = d_world_split.keys()  # Tuples of (facet_row_value, facet_col_value)
        for facet_key in keys:

            # Compute the kde
            row_value, col_value = facet_key  # split the tuple
            df_facet = d_world_split[facet_key]
            kde_grid, kde_extents = KDEItem.get_tile_fskde(
                zoom=tile_request.tile_zoom,
                x=tile_request.tile_x,
                y=tile_request.tile_y,
                tile_size=tile_request.tile_size,
                df_world=df_facet,
                use_zscore=False,
                world_x=world_x,
                world_y=world_y,
                world_value=world_value,
                adjust_bw=tile_request.kdebw
            )

            # Init KDEItem
            # (also computes .kde_norm)
            kde_item = KDEItem(
                kde_abs=kde_grid,
                facet_row_value=row_value,
                facet_col_value=col_value,
                kde_extents=kde_extents,
                kde_norm=None
            )

            # Store in dictionary
            kde_items[facet_key] = kde_item

        # Store in this instance if required
        if store_items:
            self.kde_items = kde_items

        return kde_items

    @staticmethod
    def assign_kde_groups(kde_items: Dict[Any, KDEItem],
                          facet_row_values: List[str],
                          facet_col_values: List[str],
                          rel_mode: KDERelativeMode,
                          rel_selection: KDERelativeSelection,
                          rel_row_name: str = KDEGroup.NO_SELECTION,
                          rel_col_name: str = KDEGroup.NO_SELECTION,
                          set_kde_group: bool = False) -> Dict[str, KDEGroup]:
        """
        Define and assign KDE groups -in place- if set_kde_group = True.

        This is used to compute KDE relative values.
        For each kde item, the group of kdes to which it should be
        compared to is determined.

        Generally, selection is performed relative to the kde in question or
        by using a named row and/or column.

        Sets the KDEItem.kde_group property if set_kde_group is True.


        Parameters
        ----------
        kde_items : Dict[Any, KDEItem]
            The dictionary of all KDE items (without groups assigned).
            Can be self.kde_items.

        facet_row_values : List[str]
            The ordered list of facet row values, as they appear in
            the user interface.

        facet_col_values : List[str]
            The ordered list of facet column values, as they appear in
            the user interface.

        rel_mode : KDERelativeMode
            The 'relative' mode to use. For example, whether we want to
            compare against all other KDEs, all KDEs in the current row,
            all KDEs in a specific row etc. See KDERelativeMode for options.

        rel_selection : KDERelativeSelection
            The relative 'selection' method to use. For example, whether we
            want to compare against all items in the row/column, or if
            it should just be the first/last/previous item. See
            KDERelativeSelection for options.

        rel_row_name : str, optional
            Valid if rel_mode is SINGLE, ROW or COLUMN.
            The name of the row containing the KDE(s) to compare with. If NO_SELECTION,
            will use the row of the KDE item. By default, KDEGroup.NO_SELECTION.

        rel_col_name: str, optional
            Valid if rel_mode is SINGLE, ROW or COLUMN.
            The name of the column containing the KDE(s) to compare with. If NO_SELECTION,
            will use the column of the KDE item. By default, KDEGroup.NO_SELECTION.

        set_kde_group : bool (Optional)
            If True, will set each KDEItem.kde_group property with the assigned group, by
            default False.


        Raises
        ------
        ValueError
            If rel_mode is SINGLE and either rel_row_name and rel_col_name are not supplied,
            or kde.facet_row_value, kde.facet_col_value are not supplied.

        ValueError
            If an unsupported KDERelativeSelection value is supplied.

        Returns
        -------
        Dict
            Dictionary of groups for the selected kde_items, where key=composite group key
            and value is the KDEGroup instance.

        If set_kde_group is True, sets the KDEItem.kde_group property for each item kde_items.
        """

        # Init
        kde_groups: Dict[str, KDEGroup] = dict()
        kde_items_list: List[KDEItem] = [v for k, v in kde_items.items()]
        group_row_value: Union[str, None] = KDEGroup.NO_SELECTION
        group_col_value: Union[str, None] = KDEGroup.NO_SELECTION

        # Pass 1
        # Depends on truthy values for rel_row_name and rel_col_name
        # ('' = Falsey)
        for kde in kde_items_list:

            # Create the group definition for each KDE
            if rel_mode == KDERelativeMode.SINGLE:
                # Single, named kde
                if rel_row_name and rel_col_name:
                    group_row_value = rel_row_name
                    group_col_value = rel_col_name
                elif kde.facet_row_value and kde.facet_col_value:
                    # self
                    # We are looping through each kde item...
                    # if no rel_row_name and rel_col_name supplied,
                    # then kde_facet_value will be this KDE.
                    # If so, set group values to None to use
                    # zeros grid.
                    group_row_value = None
                    group_col_value = None
                else:
                    raise ValueError(
                        'For mode KDERelativeMode.SINGLE, \
                        either both rel_row_name and rel_col_name must be \
                        provided, or kde.facet_row_value and kde_facet_col_value \
                        must be populated.'
                    )

            if rel_mode == KDERelativeMode.ALL:
                # All kdes (all rows and columns)
                if rel_selection == KDERelativeSelection.ALL:
                    group_row_value = KDEGroup.NO_SELECTION
                    group_col_value = KDEGroup.NO_SELECTION
                elif rel_selection == KDERelativeSelection.FIRST:
                    group_row_value = facet_row_values[0]
                    group_col_value = facet_col_values[0]
                elif rel_selection == KDERelativeSelection.LAST:
                    group_row_value = facet_row_values[-1]
                    group_col_value = facet_col_values[-1]
                else:
                    raise ValueError(
                        'Unsupported KDERelativeSelection value for mode.'
                    )

            elif rel_mode == KDERelativeMode.ROW:
                # Named row if supplied, otherwise row of kde item
                group_row_value = rel_row_name if rel_row_name else kde.facet_row_value

                # Column value.
                # Get all, first, last, or previous item in row.
                if rel_selection == KDERelativeSelection.ALL:
                    group_col_value = KDEGroup.NO_SELECTION
                elif rel_selection == KDERelativeSelection.FIRST:
                    group_col_value = facet_col_values[0]
                elif rel_selection == KDERelativeSelection.LAST:
                    group_col_value = facet_col_values[-1]
                elif rel_selection == KDERelativeSelection.PREVIOUS:
                    cur_col_index = facet_col_values.index(kde.facet_col_value)
                    rel_col_index = max(cur_col_index - 1, 0)
                    group_col_value = facet_col_values[rel_col_index]
                else:
                    raise ValueError(
                        'Unsupported KDERelativeSelection value for mode.'
                    )

            elif rel_mode == KDERelativeMode.COLUMN:
                # Named column if supplied, otherwise column of kde item
                group_col_value = rel_col_name if rel_col_name else kde.facet_col_value

                # Row value.
                # Get all, first, last, or previous item in column.
                if rel_selection == KDERelativeSelection.ALL:
                    group_row_value = KDEGroup.NO_SELECTION
                elif rel_selection == KDERelativeSelection.FIRST:
                    group_row_value = facet_row_values[0]
                elif rel_selection == KDERelativeSelection.LAST:
                    group_row_value = facet_row_values[-1]
                elif rel_selection == KDERelativeSelection.PREVIOUS:
                    cur_row_index = facet_row_values.index(kde.facet_row_value)
                    rel_row_index = max(cur_row_index - 1, 0)
                    group_row_value = facet_row_values[rel_row_index]
                else:
                    raise ValueError(
                        'Unsupported KDERelativeSelection value for mode.'
                    )

            # Use existing group instance or create a new one
            group_key = KDEGroup.get_group_key(
                group_row_value, group_col_value)
            if group_key in kde_groups:
                kde_group = kde_groups[group_key]
            else:
                # KDEGroup filters kde_items_list on __init__
                kde_group = KDEGroup(
                    kde_items=kde_items_list,
                    facet_row_value=group_row_value,
                    facet_col_value=group_col_value
                )
                kde_groups[group_key] = kde_group

            # Store the group instance with the kde item
            if set_kde_group:
                kde.kde_group = kde_group

        # Return all the groups used
        return kde_groups

    @staticmethod
    def compute_group_kdes(kde_groups: Dict[str, KDEGroup]):
        """Compute each group KDE -in place-"""
        for _, kde_group in kde_groups.items():
            kde_group.compute_group_kde()

    @staticmethod
    def compute_kde_diff_similar(kde_items: Dict[Any, KDEItem]):
        """Compute KDE diff and similar -in place-"""
        for _, kde_item in kde_items.items():
            kde_item.compute_diff(
                kde_norm=kde_item.kde_norm,
                relative_kde=kde_item.kde_group.group_kde,
                store_diff=True
            )
            kde_item.compute_similar(
                kde_norm=kde_item.kde_norm,
                relative_kde=kde_item.kde_group.group_kde,
                store_similar=True
            )

    def compute_kde_diff_similar_minmax(self, kde_items: Dict[Any, KDEItem], store: bool = False, diff_zero_center=True) -> Tuple[float, float, float, float]:
        """
        Compute the range of kde_diff and kde_similar properties in the KDETileSet.

        kde_diff should be in the range -1.0 to +1.0 if relative to other KDEs other than self,
        otherwise in the range 0.0 to 1.0 if relative to self. 

        When rendering and diff_zero_center is True, we want zero to be in the middle of the range, so
        the largest range above or below zero is used. If diff_zero_center is False, will use
        zero as the floor.

        kde_similar should be in the range 0.0 to +1.0, with +1.0 being identical.

        Parameters
        ----------
        kde_items : Dict[Any, KDEItem]
            The dictionary of KDEItems where kde_diff and kde_similar properties have been computed.
            self.kde_items may be used.

        store : bool (Optional)
            True to store the result in this KDETileSet's kde_diff_similar_minmax property.
            This will also set the kde_diff_similar_minmax_computed property to True.
            If it is already True, kde_diff_similar_minmax will still be overwritten.

        diff_zero_center : bool (Optional)
            True to center the kde_diff values around zero, otherwise False
            to have zero base / floor of zero. By default, True.


        Returns
        -------
        Tuple[float, float, float, float]
            [0-1]: Diff min-max range. Zero will be in the center of the 
                range if diff_zero_center is True, otherwise min will be zero.
            [2-3]: Similar min-max range.
        """

        zero_or_below = 0.
        zero_or_above = 0.

        similar_min = 0.
        similar_max = 0.

        for _, kde_item in kde_items.items():
            if kde_item.kde_diff is not None:
                zero_or_below = min(zero_or_below, kde_item.kde_diff.min())
                zero_or_above = max(zero_or_above, kde_item.kde_diff.max())
            else:
                raise Exception('kde_diff has not been computed.')

            if kde_item.kde_similar is not None:
                # max, as only want min to move up the scale.
                # should always be between 0.0 and 1.0.
                similar_min = max(similar_min, kde_item.kde_similar.min())
                similar_max = max(similar_max, kde_item.kde_similar.max())
            else:
                raise Exception('kde_similar has not been computed.')

        # For relative diffs, will be [-1.0, 1.0]. For self relative, will be [0.0, 1.0]
        # and diff_largest_range will always be zero_or_above.
        # Zero center for relative diffs, zero floor for self.
        diff_largest_range = max(abs(zero_or_below), zero_or_above)
        final_range = (-diff_largest_range if diff_zero_center else 0.0,
                       diff_largest_range, similar_min, similar_max)

        # Store
        if store:
            self.kde_diff_similar_minmax = final_range
            self.kde_diff_similar_minmax_computed = True

        return final_range


class KDEManager():

    def __init__(self, tileset_cache_size=5):
        self._tileset_cache: Cache = Cache(cache_size=tileset_cache_size)

    def get_tileset(self,
                    tile_request: TileRequest,
                    d_world_split: Dict[Any, pd.DataFrame],
                    facet_row_values: List[str],
                    facet_col_values: List[str],
                    world_x='x',
                    world_y='y',
                    world_value='value1') -> Tuple[KDETileSet, bool]:
        """
        Retrieves a KDETileSet instance, which will contain the KDEs for -all-
        facets, for the selected query and tile coordinates/zoom.

        Parameters
        ----------
        tile_request : TileRequest
            The requested tile. facet_row_value and facet_col_value will be ignored.

        d_world_split : Dict[Any, pd.DataFrame]
            The data for the current query, split by facet.
            Key=(facet_row_value, facet_col_value).

        facet_row_values : List[str]
            The ordered list of facet row values.

        facet_col_values : List[str]
            The ordered list of facet column values.

        world_x : str, optional
            The DataFrame column containing the x coordinate, by default 'x'.

        world_y : str, optional
            The DataFrame column containing the y coordinate, by default 'y'.

        world_value : str, optional
            The DataFrame column containing the measure value, by default 'value1'.

        Returns
        -------
        Tuple[KDETileSet, bool]
            The KDETileSet instance, and whether the KDETileSet was cached (True)
            or not (False).

        Raises
        ------
        ValueError
            Raises an error if the TileRequest.tile_type property is not a KDE type.
        """

        # Check tile type
        if tile_request.tile_type != 'KDE-DIFF' and tile_request.tile_type != 'KDE-SIMILAR':
            raise ValueError(
                f'Unsupported tile_type: {tile_request.tile_type}')

        # Get the key for the TileSet
        tileset_key = KDEManager.get_tileset_key(tile_request)

        # Get the TileSet if we have it in the cache
        tileset: KDETileSet = self._tileset_cache.get(
            tileset_key, if_not_exists=None)
        if tileset is not None:
            return tileset, True

        # Create a new TileSet instance
        tileset = KDETileSet(key=tileset_key)

        # Compute KDEItems
        # Sets KDEItem.kde_abs and KDEItem.kde_norm properties.
        kde_items = tileset.compute_kde_abs_norm(
            d_world_split=d_world_split,
            world_x=world_x,
            world_y=world_y,
            world_value=world_value,
            tile_request=tile_request,
            store_items=True  # Sets TileSet.kde_items property
        )

        # Get the KDEGroup for each item
        # (for computing relative values).
        # Sets KDEItem.kde_group property.
        kde_groups = tileset.assign_kde_groups(
            kde_items=kde_items,
            facet_row_values=facet_row_values,
            facet_col_values=facet_col_values,
            rel_mode=tile_request.kde_rel_mode,
            rel_selection=tile_request.kde_rel_selection,
            rel_row_name=tile_request.kde_rel_row_name,
            rel_col_name=tile_request.kde_rel_col_name,
            set_kde_group=True  # Sets kde_item.kde_group property
        )

        # Compute the KDEGroup KDEs in place (e.g. medians)
        # Sets KDEGroup.group_kde property.
        tileset.compute_group_kdes(kde_groups)

        # Compute the diff between each item's KDE and its group KDE in place.
        # Sets KDEItem.kde_diff and KDEItem.kde_similar property.
        tileset.compute_kde_diff_similar(kde_items)

        # Compute the kde_diff and kde_similar min-max range (for rendering)
        # Check first item to see if using self relative mode
        # (If so, don't zero center to use full range of colour map).
        is_self_group = False
        if len(tileset.kde_items) > 0:
            first_kde_item = next(iter(tileset.kde_items.items()))[1]
            is_self_group = first_kde_item.kde_group.is_self_group
        tileset.compute_kde_diff_similar_minmax(
            kde_items=tileset.kde_items,
            store=True,
            diff_zero_center=not is_self_group
        )

        # Store the TileSet in the cache
        self._tileset_cache.store(tileset_key, tileset)

        # Return it
        return tileset, False

    @staticmethod
    def get_tileset_key(tile_request: TileRequest):
        """
        Build a composite key for a TileSet.

        This should contain any property that will 
        affect the computation of the KDE tile images.
        Certain properties like brightness are exempt,
        as they can be handled by tile.array_to_image(), and don't
        require re-computation of the KDE (i.e.
        KDE grid values remain the same, only the rendered image changes).

        Parameters
        ----------
        tile_request : TileRequest
            The Tile Request instance.

        Returns
        -------
        str
            The composite key for initialising a TileSet.
        """

        KEY_DELIM = '|'
        KEY_PREFIX = 'k'

        key_parts = [
            KEY_PREFIX,
            tile_request.query_hash,
            tile_request.tile_x,
            tile_request.tile_y,
            tile_request.tile_zoom,
            tile_request.kdebw,
            tile_request.tile_size,
            tile_request.kde_rel_mode.name,  # KDERelativeMode enum name
            tile_request.kde_rel_selection.name,  # KDERelativeSelection enum name
            tile_request.kde_rel_row_name,
            tile_request.kde_rel_col_name
        ]

        key = KEY_DELIM.join([str(x) for x in key_parts])

        return key
