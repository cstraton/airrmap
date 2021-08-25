import unittest
import random
import pandas as pd
import numpy as np
from PIL import Image

from airrmap.application.tile import TileHelper

# %%


class TestTileHelper(unittest.TestCase):

    def setUp(self):  # Note uppercase U

        self.num_classes = 8
        self.redundancy = 2
        self.record_count = 2500
        self.tile_size = 256
        records = []
        for i in range(self.record_count):
            records.append(dict(name=str(i),
                                x=random.uniform(0, 256),
                                y=random.uniform(0, 256),
                                redundancy=self.redundancy,
                                class_name=random.randint(0, self.num_classes - 1)))
        self.df = pd.DataFrame().from_records(records)
        # ray.init() # Init ray

    def tearDown(self):
        # ray.shutdown()
        pass

    def test_tile_to_world(self):
        self.assertEqual(TileHelper.tile_to_world(0, 0, 0, 256),
                         ((0, 0), (256, 256)), 'Zoom 0, single tile')
        self.assertEqual(TileHelper.tile_to_world(1, 1, 1, 256),
                         ((128, 128), (256, 256)), 'Zoom 1, 4 tiles')
        self.assertEqual(TileHelper.tile_to_world(1, 1, 1, 128),
                         ((64, 64), (128, 128)), 'Zoom 1, 4 tiles, different tile size')
        self.assertEqual(TileHelper.tile_to_world(1, 1, 0, 256),
                         ((128, 0), (256, 128)), 'Zoom 1, 4 tiles, different xy')

    def test_get_tile_binned_lessbins(self):
        abinned = TileHelper.get_tile_binned(zoom=0, x=0, y=0, tile_size=256, num_bins=16,
                                             df_world=self.df, world_x='x', world_y='y',
                                             world_value='redundancy', statistic='sum')
        self.assertEqual(abinned.shape, (16, 16))

    def test_get_tile_binned_noclass(self):
        abinned = TileHelper.get_tile_binned(zoom=0, x=0, y=0, tile_size=256, num_bins=256,
                                             df_world=self.df, world_x='x', world_y='y',
                                             world_value='redundancy', statistic='sum')
        self.assertEqual(abinned.shape, (256, 256))
        self.assertEqual(abinned.sum(), self.record_count * self.redundancy)

    def test_get_tile_binned_withclass(self):
        abinned = TileHelper.get_tile_binned(zoom=0, x=0, y=0, tile_size=256, num_bins=256,
                                             df_world=self.df, world_x='x', world_y='y',
                                             world_value='redundancy', statistic='sum',
                                             world_class='class_name',
                                             num_classes=self.num_classes)
        self.assertEqual(abinned.shape, (256, 256, self.num_classes))
        self.assertEqual(abinned.sum(), self.record_count * self.redundancy)

    def test_array_to_image(self):
        # Convert numpy array to an image

        # Create random grid
        grid = np.random.rand(256, 256)

        # Convert to image
        im = TileHelper.array_to_image(
            grid,
            brightness_multiplier=2.,
            source_value_min=0.0,
            source_value_max=110.0,
            range_min=0.25,
            range_max=0.75,
            cmap='twilight_shifted')
        self.assertIsInstance(im, Image.Image, 'Image should be returned.')

    def test_get_tile_image_noclass(self):
        binned_data = np.random.uniform(low=0, high=255, size=(256, 256))
        img = TileHelper.get_tile_image(
            binned_data,
            value1_min=1,
            value1_max=100)
        self.assertIsInstance(img, Image.Image)

    def test_get_tile_image_withclass(self):
        binned_data = np.random.uniform(low=0, high=255, size=(256, 256, 3))
        class_rgb = np.array([(30, 20, 10), (50, 40, 30), (50, 21, 22)])
        img = TileHelper.get_tile_image(
            binned_data,
            value1_min=1,
            value1_max=100,
            class_rgb=class_rgb)
        self.assertIsInstance(img, Image.Image)

    def test_get_circle_mask(self):
        mask = TileHelper.get_circle_mask(256, 128)
        self.assertIsInstance(mask, Image.Image)

    def test_get_scaled_brightness(self):
        max_power2 = 12
        binned_values = np.random.uniform(
            low=0, high=max_power2, size=(256, 256, 1))
        result = TileHelper.get_scaled_brightness(
            binned_values,
            zoom_level=0,
            max_power2=max_power2,
            smooth=False)
        self.assertIsInstance(result, np.ndarray)
        self.assertGreater(
            result.sum(),
            0,
            'Sum of values should be greater than zero.'
        )


# %%
def get_test_data(num_classes):
    """Generate a test dataset"""
    records = []
    for i in range(10000):
        records.append(dict(name=str(i),
                            x=random.normalvariate(mu=128, sigma=40),
                            y=random.normalvariate(mu=128, sigma=40),
                            redundancy=2,
                            class_id=random.randint(0, num_classes - 1)))

    df_world = pd.DataFrame.from_dict(records)
    records = None
    return df_world


# %%
if __name__ == '__main__':
    # pass
    unittest.main(argv=[''], verbosity=2, exit=False)
