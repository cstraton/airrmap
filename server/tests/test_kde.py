# %% imports
import random
import math
import unittest
import pandas as pd
from airrmap.shared.models import TileRequest
from airrmap.application.kde import *


class TestKDEItem(unittest.TestCase):

    def setUp(self):
        self.num_classes = 8
        self.redundancy = 2
        self.record_count = 10000
        self.tile_size = 256
        records = []
        for i in range(self.record_count):
            records.append(dict(name=str(i),
                                x=random.uniform(0, 256),
                                y=random.uniform(0, 256),
                                redundancy=self.redundancy,
                                class_name=random.randint(0, self.num_classes - 1)))
        self.df = pd.DataFrame().from_records(records)

    def test_init(self):
        x = KDEItem(
            kde_abs=np.random.rand(256, 256),
            facet_row_value='row1',
            facet_col_value='col1',
            kde_group=None,
            kde_norm=None
        )
        self.assertIsInstance(
            x,
            KDEItem,
            'KDEItem instance should be created.'
        )
        self.assertAlmostEqual(
            x.kde_norm.sum(),
            1.0,
            4,
            'Normalised kde should be automatically if kde_norm=None.'
        )
        self.assertEqual(
            x.key,
            KDEItem.get_key('row1', 'col1'),
            'Key should be set.'
        )

    def test_normalize(self):
        # Center around 0 to also test negatives
        x = np.random.rand(256, 256) - 0.5
        x_norm = KDEItem.normalize(x)
        self.assertAlmostEqual(
            x_norm.sum(),
            1.0,
            8,
            'Normalized array should sum to 1.0.'
        )

        x = np.zeros((256, 256))
        x_norm = KDEItem.normalize(x)
        self.assertIs(
            x,
            x_norm,
            "Array should be returned as is if zero-sum (can't divide by zero)."
        )

    def test_get_key(self):
        self.assertEqual(
            KDEItem.get_key('row1', 'col1'),
            'k|row1|col1',
            'Composite key should be returned.'
        )
        self.assertEqual(
            KDEItem.get_key(None, None),
            f'{KDEItem.KEY_PREFIX}__NONE{KDEItem.KEY_DELIM}__NONE',
            'Correct composite key should be returned for None values.'
        )

    # def test_get_tile_kde(self):
        # Render tile using kernel density estimate
        # img = TileHelper.get_tile_kde(
        #    zoom=0, x=0, y=0, tile_size=256, num_bins=256,
        #    df_world=self.df, world_x='x', world_y='y',
        #    world_value='redundancy',
        #    image_size=256
        # )
        #self.assertIsInstance(img, Image.Image)

    def test_get_tile_fskde(self):
        # Render tile using kernel density estimate with faststats kde
        # NOTE: Slow due to use of random data, faster on real-world.
        kde_grid, kde_extents = KDEItem.get_tile_fskde(
            zoom=0, x=0, y=0, tile_size=256,
            df_world=self.df, use_zscore=True, world_x='x', world_y='y',
            world_value='redundancy',
            adjust_bw=0.75
        )
        self.assertIsNotNone(kde_grid)
        self.assertIsNotNone(kde_extents)

    def test_interp_kde_region(self):
        # int
        tile_size = self.tile_size

        # Get kde grid
        kde_grid, kde_extents = KDEItem.get_tile_fskde(
            zoom=0, x=0, y=0, tile_size=tile_size,
            df_world=self.df, world_x='x', world_y='y',
            world_value='redundancy',
            adjust_bw=0.75
        )

        # Interpolate (note zoom=1 to get subset)
        kde_interpolated = KDEItem.interp_kde_region(
            zoom=1, x=0, y=0, tile_size=tile_size, kde_world_grid=kde_grid
        )

        self.assertIsInstance(kde_interpolated, np.ndarray,
                              'ndarray should be returned')
        self.assertEqual(kde_interpolated.shape, (tile_size, tile_size),
                         f'Interpolated kde should be of shape ({tile_size}, {tile_size})')

    def test_compute_diff(self):

        kde_abs = np.ones((256, 256))
        kde_item = KDEItem(
            kde_abs=kde_abs, facet_row_value='', facet_col_value='')

        # Init KDEItem, kde_norm will be set automatically
        kde_diff = kde_item.compute_diff(
            kde_norm=kde_item.kde_norm,
            relative_kde=kde_item.kde_norm * 2,
            store_diff=True
        )

        self.assertTrue(
            np.array_equal(kde_diff, kde_item.kde_norm -
                           (kde_item.kde_norm * 2)),
            'compute_diff() should return correct difference in arrays.'
        )

        self.assertIs(
            kde_item.kde_diff,
            kde_diff,
            'kde.kde_diff property should be set if store_diff is True.'
        )


    def test_compute_similar(self):

        kde_abs = np.ones((256, 256))
        kde_item = KDEItem(
            kde_abs=kde_abs, facet_row_value='', facet_col_value='')

        # Init KDEItem, kde_norm will be set automatically
        kde_similar = kde_item.compute_similar(
            kde_norm=kde_item.kde_norm,
            relative_kde=kde_item.kde_norm * 2,
            store_similar=True
        )

        self.assertTrue(
            np.array_equal(kde_similar, 1.0 - np.abs(kde_item.kde_norm -
                           (kde_item.kde_norm * 2))),
            'compute_similar() should return correct similarity between arrays.'
        )

        self.assertIs(
            kde_item.kde_similar,
            kde_similar,
            'kde.kde_similar property should be set if store_similar is True.'
        )


class TestKDEGroup(unittest.TestCase):
    """Filters and stores a group of KDEs"""

    def setUp(self):

        # Generate test list of KDE items.
        facet_row_values = ['row1', 'row2', 'row3']
        facet_col_values = ['col1', 'col2', 'col3']
        kde_items: List[KDEItem] = []
        for row_value, col_value in itertools.product(facet_row_values, facet_col_values):
            key = '|'.join([row_value, col_value])
            kde_item = KDEItem(
                facet_row_value=row_value,
                facet_col_value=col_value,
                kde_abs=np.random.rand(256, 256),
                kde_norm=None
            )
            kde_items.append(kde_item)
        self.kde_items = kde_items

    def test_None(self):
        kde_group = KDEGroup(
            kde_items=self.kde_items,
            facet_row_value=None,
            facet_col_value=None
        )
        first_kde_item_shape = self.kde_items[0].kde_abs.shape
        self.assertTrue(
            np.array_equal(
                 kde_group.group_kde_items[0].kde_abs,
                 np.zeros(first_kde_item_shape)
            ),
            'Group item should be grid of zeros with correct shape.'
        )
        self.assertTrue(
            kde_group.is_self_group,
            'is_self_group should be set to True.'
        )

    def test_all(self):
        kde_group = KDEGroup(
            kde_items=self.kde_items,
            facet_row_value=KDEGroup.NO_SELECTION,
            facet_col_value=KDEGroup.NO_SELECTION
        )
        self.assertListEqual(
            kde_group.group_kde_items,
            self.kde_items,
            'All items should be in group scope.'
        )
        self.assertFalse(
            kde_group.is_self_group,
            'is_self_group should be set to False.'
        )

    def test_row(self):
        kde_group = KDEGroup(
            kde_items=self.kde_items,
            facet_row_value='row1',
            facet_col_value=KDEGroup.NO_SELECTION
        )
        self.assertListEqual(
            kde_group.group_kde_items,
            [x for x in self.kde_items if x.facet_row_value == 'row1'],
            'Only row1 should be in group scope.'
        )

    def test_column(self):
        kde_group = KDEGroup(
            kde_items=self.kde_items,
            facet_row_value=KDEGroup.NO_SELECTION,
            facet_col_value='col1'
        )
        self.assertListEqual(
            kde_group.group_kde_items,
            [x for x in self.kde_items if x.facet_col_value == 'col1'],
            'Only col1 should be in group scope.'
        )

    def test_single(self):
        kde_group = KDEGroup(
            kde_items=self.kde_items,
            facet_row_value='row2',
            facet_col_value='col2'
        )
        self.assertListEqual(
            kde_group.group_kde_items,
            [x for x in self.kde_items
                if x.facet_row_value == 'row2' and x.facet_col_value == 'col2'],
            'Only single kde row2, col2 should be in group scope.'
        )

    def test_group_key(self):
        self.assertIsInstance(
            KDEGroup.get_group_key('row1', 'col1'),
            str,
            'Composite group key should be returned.'
        )
        self.assertEqual(
            KDEGroup.get_group_key(None, None),
            f'{KDEGroup.KEY_PREFIX}__NONE{KDEGroup.KEY_DELIM}__NONE',
            'Correct composite key should be returned for None values.'
        )

    def test_compute_group_kde(self):
        kde_group = KDEGroup(
            kde_items=self.kde_items,
            facet_row_value=KDEGroup.NO_SELECTION,
            facet_col_value=KDEGroup.NO_SELECTION
        )
        kde_median = kde_group.compute_group_kde()
        self.assertEqual(
            kde_median.shape,
            self.kde_items[0].kde_norm.shape,
            'Returned ndarray should be same shape as original kde.'
        )
        self.assertTrue(
            np.array_equal(
                np.median(
                    np.dstack(tuple([x.kde_norm for x in self.kde_items])),
                    axis=2
                ),
                kde_median,
            ),
            'ndarray should be the median of the original 2D arrays.'
        )


class TestKDETileSet(unittest.TestCase):

    def get_query_split_data(self) -> Tuple[Dict, List, List]:
        # Generate test KDE items, per facet
        records_per_facet = 100
        facet_row_values = ['row1', 'row2']
        facet_col_values = ['col1', 'col2']
        d_split_by_facet = {}
        for row_value, col_value in itertools.product(facet_row_values, facet_col_values):
            records = [
                dict(
                    facet_row=row_value,
                    facet_col=col_value,
                    # Fixed with jitter, random values slow KDE test. Same value causes single matrix error.
                    x=1. + random.random() - 0.5,
                    # Fixed with jitter, random values slow KDE test. Same value causes single matrix error.
                    y=2. + random.random() - 0.5,
                    value1=random.randint(0, 10)
                )
                for i in range(records_per_facet)
            ]
            df_records = pd.DataFrame.from_records(records)
            key = (row_value, col_value)
            d_split_by_facet[key] = df_records

        return d_split_by_facet, facet_row_values, facet_col_values

    def get_test_kde_items(self) -> Tuple[Dict, List, List]:
        # Generate test KDEItems and facet row and column values
        facet_row_values = ['row1', 'row2', 'row3']
        facet_col_values = ['col1', 'col2']

        # Build square facets of random 2D data
        kde_items: Dict[Any, KDEItem] = {}
        for row_value, col_value in itertools.product(facet_row_values, facet_col_values):
            kde_abs = np.random.rand(256, 256)
            kde_item = KDEItem(
                kde_abs=kde_abs,
                facet_row_value=row_value,
                facet_col_value=col_value
            )
            key = (row_value, col_value)
            kde_items[key] = kde_item

        return kde_items, facet_row_values, facet_col_values

    def test_compute_kde_abs_norm(self):

        # Get test data
        d_split_by_facet, facet_row_values, facet_col_values = self.get_query_split_data()

        # Init
        x = KDETileSet()
        tile_request = TileRequest.get_dummy_instance()
        tile_request.kdebw = 0.75
        tile_request.tile_size = 256
        tile_request.tile_zoom = 0
        tile_request.tile_x = 0
        tile_request.tile_y = 0
        num_facets = len(d_split_by_facet)

        # Compute the kdes
        result = x.compute_kde_abs_norm(
            d_world_split=d_split_by_facet,
            world_x='x',
            world_y='y',
            world_value='value1',
            tile_request=tile_request,
            store_items=True
        )

        self.assertIsInstance(
            result,
            dict,
            'compute_kde_abs() should return a dictionary.'
        )
        self.assertEqual(
            len(result),
            num_facets,
            'compute_kde_abs() result length should be number of facets.'
        )
        self.assertEqual(
            len(x.kde_items),
            num_facets,
            'TileSet kde_items property should be set when store_items is True.'
        )

        for k, v in result.items():
            self.assertIsInstance(
                v,
                KDEItem,
                'Each item should be a KDEItem instance.'
            )
            self.assertIsInstance(
                v.kde_abs,
                np.ndarray,
                'Each KDEItem should have its kde_abs property set to an kde ndarray.'
            )
            self.assertIsInstance(
                v.kde_norm,
                np.ndarray,
                'Each KDEItem should have its kde_norm property set to an kde ndarray.'
            )

    def test_assign_kde_groups(self):

        o = KDETileSet()
        kde_items, facet_row_values, facet_col_values = self.get_test_kde_items()

        # NOTE: Assignment to kde_items happens in-place.
    
    
        # ALL mode
        kde_groups = o.assign_kde_groups(
            kde_items=kde_items,
            facet_row_values=facet_row_values,
            facet_col_values=facet_col_values,
            rel_mode=KDERelativeMode.ALL,
            rel_selection=KDERelativeSelection.ALL,
            rel_row_name=KDEGroup.NO_SELECTION,
            rel_col_name=KDEGroup.NO_SELECTION,
            set_kde_group=False
        )
        self.assertEqual(
            len(kde_groups),
            1,
            'One group for all should be returned.'
        )

        # ROW mode (self)
        kde_groups = o.assign_kde_groups(
            kde_items=kde_items,
            facet_row_values=facet_row_values,
            facet_col_values=facet_col_values,
            rel_mode=KDERelativeMode.ROW,
            rel_selection=KDERelativeSelection.ALL,
            rel_row_name=KDEGroup.NO_SELECTION,
            rel_col_name=KDEGroup.NO_SELECTION,
            set_kde_group=False
        )
        self.assertEqual(
            len(kde_groups), 3), 'Three groups should be returned (one for each row).'

        # COL mode (self)
        kde_groups = o.assign_kde_groups(
            kde_items=kde_items,
            facet_row_values=facet_row_values,
            facet_col_values=facet_col_values,
            rel_mode=KDERelativeMode.COLUMN,
            rel_selection=KDERelativeSelection.ALL,
            rel_row_name=KDEGroup.NO_SELECTION,
            rel_col_name=KDEGroup.NO_SELECTION,
            set_kde_group=False
        )
        self.assertEqual(
            len(kde_groups), 2), 'Two groups should be returned (one for each column).'

        # SINGLE Mode (self / ALL)
        kde_groups = o.assign_kde_groups(
            kde_items=kde_items,
            facet_row_values=facet_row_values,
            facet_col_values=facet_col_values,
            rel_mode=KDERelativeMode.SINGLE,
            rel_selection=KDERelativeSelection.ALL,
            rel_row_name=KDEGroup.NO_SELECTION,
            rel_col_name=KDEGroup.NO_SELECTION,
            set_kde_group=False
        )
        self.assertEqual(
            len(kde_groups),
            1,
            'One group for all facets should be returned (zero grid).'
        )

        first_kde_item_shape: KDEItem = list(kde_items.values())[0].kde_abs.shape
        last_group_item: KDEItem = kde_groups[list(kde_groups.keys())[-1]].group_kde_items[-1]
        self.assertTrue(
            np.array_equal(
                np.zeros(first_kde_item_shape),
                last_group_item.kde_abs,
            ),
            'Group KDEs should be zero grids if SINGLE mode and ALL selection (self).'
        )

        self.assertEqual(
            last_group_item.facet_row_value,
            None,
            "Last group item's row value should be None."
        )
        self.assertEqual(
            last_group_item.facet_col_value,
            None,
            "Last group item's column value should be None"
        )

        # SINGLE Mode (named)
        kde_groups = o.assign_kde_groups(
            kde_items=kde_items,
            facet_row_values=facet_row_values,
            facet_col_values=facet_col_values,
            rel_mode=KDERelativeMode.SINGLE,
            rel_selection=KDERelativeSelection.ALL,
            rel_row_name='row3',
            rel_col_name='col2',
            set_kde_group=False
        )
        self.assertEqual(
            len(kde_groups),
            1,
            'One group should be returned for the selected row and column.'
        )
        last_item = kde_groups[list(kde_groups.keys())[0]].group_kde_items[-1]
        self.assertEqual(
            last_item.facet_row_value,
            'row3',
            'KDE item should be for the selected named row.'
        )
        self.assertEqual(
            last_item.facet_col_value,
            'col2',
            'KDE item should be for the selected named column.'
        )

        # ALL Mode (named), FIRST
        kde_groups = o.assign_kde_groups(
            kde_items=kde_items,
            facet_row_values=facet_row_values,
            facet_col_values=facet_col_values,
            rel_mode=KDERelativeMode.ALL,
            rel_selection=KDERelativeSelection.FIRST,
            rel_row_name=KDEGroup.NO_SELECTION,
            rel_col_name=KDEGroup.NO_SELECTION,
            set_kde_group=False
        )
        self.assertEqual(
            len(kde_groups),
            1,
            'One group should be returned (first item).'
        )

        # ALL mode, set KDE groups
        kde_groups = o.assign_kde_groups(
            kde_items=kde_items,
            facet_row_values=facet_row_values,
            facet_col_values=facet_col_values,
            rel_mode=KDERelativeMode.ALL,
            rel_selection=KDERelativeSelection.ALL,
            rel_row_name=KDEGroup.NO_SELECTION,
            rel_col_name=KDEGroup.NO_SELECTION,
            set_kde_group=True
        )
        only_kde_group = kde_groups[list(kde_groups.keys())[0]]
        for k, kde_item in kde_items.items():
            self.assertIs(
                kde_item.kde_group,
                only_kde_group,
                'All kde_items should have kde_group property set to correct KDEGroup instance.'
            )

    def test_compute_group_kdes(self):

        # Init
        tileset = KDETileSet()
        kde_items, facet_row_values, facet_col_values = self.get_test_kde_items()
        kde_groups = tileset.assign_kde_groups(
            kde_items=kde_items,
            facet_row_values=facet_row_values,
            facet_col_values=facet_col_values,
            rel_mode=KDERelativeMode.ALL,
            rel_selection=KDERelativeSelection.ALL,
            set_kde_group=True
        )

        # Compute group kdes
        tileset.compute_group_kdes(kde_groups)

        # Check KDE for each group is compute
        for _, kde_group in kde_groups.items():
            self.assertTrue(
                kde_group.is_group_kde_computed,
                'compute_group_kdes() should set is_group_kde_computed property to True.'
            )
            self.assertIsInstance(
                kde_group.group_kde,
                np.ndarray,
                'kde_group.group_kde property should be set to a kde grid (ndarray)'
            )

    def test_compute_kde_diff_similar(self):
        # Init
        tileset = KDETileSet()
        kde_items, facet_row_values, facet_col_values = self.get_test_kde_items()
        kde_groups = tileset.assign_kde_groups(
            kde_items=kde_items,
            facet_row_values=facet_row_values,
            facet_col_values=facet_col_values,
            rel_mode=KDERelativeMode.ALL,
            rel_selection=KDERelativeSelection.ALL,
            set_kde_group=True
        )
        tileset.compute_group_kdes(kde_groups)

        tileset.compute_kde_diff_similar(kde_items)
        for _, kde_item in kde_items.items():
            self.assertIsInstance(
                kde_item.kde_diff,
                np.ndarray,
                'compute_kde_diff_similar() should set the kde_item.kde_diff property.'
            )
            self.assertIsInstance(
                kde_item.kde_similar,
                np.ndarray,
                'compute_kde_diff_similar() should set the kde_item.kde_similar property.'
            )

    def test_compute_kde_diff_similar_minmax(self):
        # Init
        tileset = KDETileSet()
        kde_items, facet_row_values, facet_col_values = self.get_test_kde_items()
        kde_groups = tileset.assign_kde_groups(
            kde_items=kde_items,
            facet_row_values=facet_row_values,
            facet_col_values=facet_col_values,
            rel_mode=KDERelativeMode.ALL,
            rel_selection=KDERelativeSelection.ALL,
            set_kde_group=True
        )
        tileset.compute_group_kdes(kde_groups)

        # Compute the range of the kde.diff and kde.similar properties
        kde_diff_similar_range = tileset.compute_kde_diff_similar_minmax(
            tileset.kde_items, store=True)

        # Check result
        # Should be (diff min, diff max, similar min, similar max).
        self.assertTrue(
            isinstance(kde_diff_similar_range, tuple) and len(
                kde_diff_similar_range) == 4,
            'compute_kde_diff_similar_minmax() should return a tuple of length 4.',
        )
        self.assertIs(
            tileset.kde_diff_similar_minmax,
            kde_diff_similar_range,
            'KDETileSet.kde_diff_similar_minmax property should be set.'
        )
        self.assertTrue(
            tileset.kde_diff_similar_minmax_computed,
            'KDETileSet.kde_diff_similar_minmax_computed property should be set to True.'
        )
        self.assertEqual(
            abs(kde_diff_similar_range[0]),
            kde_diff_similar_range[1],
            'Diff range min-max should be equal opposites, so that zero lies in the center.'
        )
        self.assertTrue(
            kde_diff_similar_range[0] >= -
            1. and kde_diff_similar_range[0] <= 0.,
            'Diff minimum should be between -1.0 and 0.0 inclusive.'
        )
        self.assertTrue(
            kde_diff_similar_range[1] >= 0. and kde_diff_similar_range[1] <= 1.,
            'Diff maximum should be between 0.0 and 1.0 inclusive.'
        )

        self.assertTrue(
            kde_diff_similar_range[2] >= 0. and kde_diff_similar_range[2] <= 1.,
            'Similar minimum should be between 0.0 and 1.0 inclusive.'
        )
        self.assertTrue(
            kde_diff_similar_range[3] >= 0. and kde_diff_similar_range[3] <= 1.,
            'Similar maximum should be between 0.0 and 1.0 inclusive.'
        )
        self.assertTrue(
            kde_diff_similar_range[3] >= kde_diff_similar_range[2],
            'Similar maximum should be greater than or equal to similar minimum.'
        )

        # Test zero floor for self relative
        kde_diff_similar_range = tileset.compute_kde_diff_similar_minmax(
            tileset.kde_items, 
            store=True, 
            diff_zero_center=False
        )
        self.assertEqual(
            kde_diff_similar_range[0],
            0.0,
            'Diff minimum should be zero if diff_zero_center is False.'
        )
        self.assertTrue(
            kde_diff_similar_range[1] >= 0. and kde_diff_similar_range[1] <= 1.,
            'Diff maximum should be bewteen 0.0 and 1.0 inclusive if diff_zero_center is False.'
        )
        



class TestKDEManager(unittest.TestCase):

    def get_query_split_data(self) -> Tuple[Dict, List, List]:
        # Generate test KDE items, per facet
        records_per_facet = 100
        facet_row_values = ['row1', 'row2']
        facet_col_values = ['col1', 'col2']
        d_split_by_facet = {}
        for row_value, col_value in itertools.product(facet_row_values, facet_col_values):
            records = [
                dict(
                    facet_row=row_value,
                    facet_col=col_value,
                    # Fixed with jitter, random values slow KDE test. Same value causes single matrix error.
                    x=1. + random.random() - 0.5,
                    # Fixed with jitter, random values slow KDE test. Same value causes single matrix error.
                    y=2. + random.random() - 0.5,
                    value1=random.randint(0, 10)
                )
                for i in range(records_per_facet)
            ]
            df_records = pd.DataFrame.from_records(records)
            key = (row_value, col_value)
            d_split_by_facet[key] = df_records

        return d_split_by_facet, facet_row_values, facet_col_values

    def test__init__(self):
        self.assertIsInstance(
            KDEManager(),
            KDEManager,
            'KDEManager() should initialise successfully.'
        )

    def test_get_tileset_key(self):

        tile_request = TileRequest.get_dummy_instance()
        key = KDEManager.get_tileset_key(tile_request)
        self.assertIsInstance(
            key,
            str,
            'get_tileset_key() should return a string.'
        )
        self.assertGreater(
            len(key),
            10,
            'get_tileset_key() should return a non-zero length composite key.'
        )

    def test_get_kde_tile(self):

        # Get tile request
        tile_request = TileRequest.get_dummy_instance()
        tile_request.tile_type = 'KDE-DIFF'

        # Get the data, split by facet
        d_world_split, facet_row_values, facet_col_values = self.get_query_split_data()

        # Init KDEManager
        manager = KDEManager()

        # Generate the tileset
        tileset, cached = manager.get_tileset(
            tile_request=tile_request,
            d_world_split=d_world_split,
            facet_row_values=facet_row_values,
            facet_col_values=facet_col_values
        )

        self.assertEqual(
            manager._tileset_cache.count(),
            1,
            'KDETileSet should be stored in the cache.'
        )

        self.assertIsInstance(
            tileset,
            KDETileSet,
            'get_tileset() should return a KDETileSet instance.'
        )
        self.assertEqual(
            len(tileset.kde_items),
            len(d_world_split),
            'KDETileSet items should be set.'
        )
        self.assertFalse(
            cached,
            'KDETileSet should not have been cached on first call.'
        )

        # Test the cache by making another call
        tileset, cached = manager.get_tileset(
            tile_request=tile_request,
            d_world_split=d_world_split,
            facet_row_values=facet_row_values,
            facet_col_values=facet_col_values
        )
        self.assertTrue(
            cached,
            'KDETileSet should be retrieved from the cache on the second call.'
        )


# %%
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
