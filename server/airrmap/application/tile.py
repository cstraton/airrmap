# Tile and 2D histogram functionality

# tests > test_tile_helper.py

# %% imports
import seaborn as sns
import unittest
import pandas as pd
import random
import numpy as np
import json
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage.filters import gaussian_filter  # blur filter
from scipy import stats
from matplotlib import cm
from typing import List, Tuple, Optional, Any
from numba import jit, njit


# %%


class TileHelper:

    """Helper functions for working with tiles."""

    def __init__(self):
        """Construct a new TileHelper instance."""
        pass

    @staticmethod
    def tile_to_world(zoom: int, x: int, y: int, tile_size: int) -> \
            Tuple[Tuple[float, float], Tuple[float, float]]:
        """Compute world boundary from single tile coordinates

        Tile size should match world space (e.g. 256x256).
        Origin 0,0 is top left, higher y values are towards the bottom.

        Args:
            zoom (int): Zoom level (0 = Single tile covers whole world)
            x (int): Tile x coordinate. 0 = left.
            y (int): Tile y coordinate. 0 = top.
            tile_size (int): Size in pixels of a tile (e.g. 256x256px).
                This should match the world space [0-256], [0-256].

        Returns:
            Tuple: ((x_from, y_from), (x_to, y_to)) world coordinates

        .. _Derived from Google Map and Tile Coordinates Guide:
            https://developers.google.com/maps/documentation/javascript/coordinates
        """

        a = tile_size / (2 ** zoom)  # compute once

        x_from = x * a
        x_to = (x + 1) * a

        y_from = y * a
        y_to = (y + 1) * a

        return ((x_from, y_from), (x_to, y_to))

    @staticmethod
    def array_to_image(
            grid: np.ndarray,
            source_value_min,
            source_value_max,
            brightness_multiplier: float = 1.0,
            range_min: float = 0.0,
            range_max: float = 1.0,
            invert_values: bool = False,
            cmap: str = 'twilight_shifted') -> Image.Image:
        """
        Convert numpy array to image and scale as required.

        Parameters
        ----------
        grid : ndArray
            The numpy grid array to convert to an image.

        source_value_min: float
            Minimum value that will be encountered in
            the overall data (not just this region).

        source_value_max: float
            Maximum value that will be encountered in
            the overall data (not just this region).

        brightness_multiplier : float
            Multiply values to increase/decrease brightness.

        range_min : float, optional
            Scale-from minimum, by default 0.0. Should
            not be less than 0.0 or higher than 1.0
            (values will be clipped if outside this range.)

        range_max : float, optional
            Scale-to maximum, by default 1.0. Should not
            be less than 0.0 or higher than 1.0
            (values will be clipped if outside this range.)

        invert_values : bool, optional
            If True, will negate the values, so 1.0 becomes -1.0 and
            vice versa (useful if we want colormaps in the opposite
            direction).

        cmap : str, optional
            The matplotlib colormap to use, by default 'twilight_shifted'.
            See https://matplotlib.org/stable/tutorials/colors/colormaps.html

        Returns
        -------
        Image.Image
            The Pillow image instance.
        """

        # Invert values if required
        # (e.g. to reverse colourmaps)
        # Matplotlib uses '_r' suffix.
        cmap_reverse = ''
        if invert_values:
            cmap_reverse = '_r'

        # Clip range values if required
        range_min = max(0.0, range_min)
        range_min = min(1.0, range_min)
        range_max = max(0.0, range_max)
        range_max = min(1.0, range_max)

        # Apply brightness multiplier.
        # Should be done before np.interp()
        # as may contain +/- values that need
        # to be scaled equally so center remains 0
        # for diverging colour map.
        # (e.g. for KDE relative values [-1,1])
        scaled_grid = grid * brightness_multiplier

        # Scale the 2D array values to range
        # No need to clip afterwards as range should be in [0,1].
        scaled_grid = np.interp(
            scaled_grid,
            (source_value_min, source_value_max),
            (range_min, range_max)
        )

        # Transpose (numpy array should have Y first for image)
        scaled_grid = np.swapaxes(scaled_grid, 0, 1)

        # Get image
        # Pillow requires integers between 0 and 255
        #im = Image.fromarray(image_array, 'L')
        im = Image.fromarray(np.uint8(cm.get_cmap(
            f'{cmap}{cmap_reverse}')(scaled_grid) * 255))
        return im

    @staticmethod
    def get_tile_binned(zoom: int, x: int, y: int,
                        tile_size: int,
                        num_bins: int,
                        df_world: pd.DataFrame,
                        world_x: str = 'x',
                        world_y: str = 'y',
                        world_value: str = 'value1',
                        statistic='sum',
                        world_class: str = None,
                        num_classes: int = None):
        """Compute binned data for given tile.

        2D binning is used by default. If world_class column is supplied,
        a 3D bin is computed initially, then the class with the largest
        value is selected.

        Args:
            zoom (int): Zoom level (0 = whole world, 1 tile)
            x (int): Tile x coordinate
            y (int): Tile y coordinate
            tile_size (int): Tile size in pixels
            num_bins (int): Number of bins to use (x and y)
            df_world (pd.DataFrame): DataFrame containing world data
            world_x (str): Column with world x coordinates
            world_y (str): Column with world y coordinates
            world_value (str): Column with value to measure
            statistic: Not implemented, always sum. (The statistic to be computed for each bin).
            world_class (str): Column to use for class of record. Class
                values should be zero-based sequential integers as they
                will be used as array indicies.
            num_classes (int): Number of possible classes (1-based).
                Required if world_class is used. Drives 3D binning / colour.


        Returns:
            ndarray: The binned data, where axis 0 = Y and axis 1 = X.
                If world_class is not supplied then this will be of shape 
                (tile_size, tile_size). If world_class is used, this will be
                of shape (tile_size, tile_size, 2), where axis 2 contains
                the highest value [0] and the class index that had that value
                [1].
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

        # Get the exact bin edges / fence posts = N+1
        # (don't derive from the range of data which varies)
        bin_edges_x = np.linspace(
            start=x_from,
            stop=x_to, num=num_bins + 1,
            endpoint=True

        )
        bin_edges_y = np.linspace(
            start=y_from,
            stop=y_to,
            num=num_bins + 1,
            endpoint=True
        )

        bin_edges_classes = None

        if num_classes is not None:
            bin_edges_classes = np.linspace(
                start=0,
                stop=num_classes,
                num=num_classes + 1,
                endpoint=True
            )

        # Bin world values to given tile size
        # TODO: Use optimized library
        binned_data: Optional[np.ndarray] = None
        if world_class is not None:

            # 3D bin: Bin values (sum) by X, Y and class
            # For each XY, get the class bin with max summed value.
            # Note: Axis 0 = Y to conform with PIL image processing.
            bin_ranges = x_from,
            binned_3d = np.histogramdd(
                df_roi[[world_y, world_x, world_class]].values,
                bins=(bin_edges_y, bin_edges_x, bin_edges_classes),
                #bins=(bins, bins, num_classes),
                density=False,
                weights=df_roi[world_value]
            )

            # SciPy alternative - allows use of a different statistic (other than sum())
            # but slower to run.
           # binned_3d = stats.binned_statistic_dd(df_roi[[world_y, world_x, world_class]].values,
           #                                      values=df_roi[world_value].values,
           #                                      statistic=statistic,
           #                                      bins=(bin_edges_y, bin_edges_x, bin_edges_classes))

            # [0] for binned data without xy edges from
            # numpy.histogramddd tuple
            # Axis 2 must be num_classes so each bin represents a class.
            # Replace Nan with 0
            binned_3d = np.nan_to_num(binned_3d[0])

            #assert binned_3d.shape == (tile_size, tile_size, num_classes)
            binned_data = binned_3d

        else:

            # 2D bin: Bin value by X and Y
            # Note: Axis 0 = Y to conform with PIL image processing.
            binned_data = np.histogram2d(
                x=df_roi[world_y],
                y=df_roi[world_x],
                bins=(bin_edges_y, bin_edges_x),
                weights=df_roi[world_value]
            )[0]

        return binned_data

    @staticmethod
    def get_test_card():
        "Generate striped test card data"

        world_size = 256

        x = list(1 for i in range(world_size))

        points = []
        class_index = -1
        rangevals = [i for i in range(256)]

        for x in rangevals:
            if int(x) % 32 == 0:
                class_index += 1
            for y in rangevals:
                points.append((x, y, class_index))

        df = pd.DataFrame(
            data=points,
            index=None,
            columns=['x', 'y', 'class_index']
        )

        df['redundancy'] = list(1 for i in range(len(df)))

        # Number of classes
        dfg = df.groupby('class_index')
        num_classes = dfg.ngroups

        # Get colours for label encodings (list of RGB tuples)
        rgb_palette = sns.mpl_palette(
            "Set2",
            num_classes,
            as_cmap=False
        )

        rgb_palette = np.array(rgb_palette)
        class_rgb = rgb_palette

        # Return
        return df, num_classes, class_rgb

    @staticmethod
    @jit()
    def get_tile_image(binned_data: np.ndarray,
                       value1_min: float,
                       value1_max: float,
                       class_rgb=None,
                       image_size: int = 256,
                       tile_mask: Image.Image = None,
                       brightness: float = 1.0) -> Image.Image:
        """Generate tile image from binned tile data

        Args:
            binned_data (ndarray): If shape (tile_size Y, tile_size X), will
                use values as brightness. If (tile_size Y, tile_size X, num_classes),
                will use the class with the highest value, and assign colours.

            value1_min (float): The minimum 'value1' in the dataset.
                Used for scaling (e.g. 0.0-1.0). Can be a negative value.

            value1_max (float): The maximum 'value1' in the dataset.
                Used for scaling (e.g. 0.0-1.0). Can be a negative value.

            class_rgb(ndarray): (N,3) RGB arrays (0.0-1.0 )for each class
                (if class is used). N should correspond with class index
                (axis 2 in binned_data).

            image_size(int): Tile size in pixels.

            tile_mask(Image.Image): Mask image to place over final tile image
                (e.g. circles). Default is None.

            brightness(float): Brightness multiplier. Default is 1.0.

        Returns:
            PIL.Image: Tile image
        """

        # TODO: Optimization

        # init
        rgb_values = None  # (Y, X, 3)
        binned_values = None  # (Y, X)

        if binned_data.ndim == 3:

            # Get class/values with highest bin sum
            max_classes = np.argmax(binned_data, axis=2)
            max_values = np.amax(binned_data, axis=2)
            binned_values = max_values

            # Get RGB for classes, shape (Y, X, 3)
            # REF: https://stackoverflow.com/questions/8188726/how-do-i-do-this-array-lookup-replace-with-numpy
            rgb_values = class_rgb[max_classes]

        elif binned_data.ndim == 2:

            binned_values = binned_data

            # Default to white - scale in a minute...
            rgb_values = np.ones(shape=(binned_data.shape[0],
                                        binned_data.shape[1], 3))

        else:
            raise Exception('Unexpected data shape: ' + str(binned_data.shape))

        # Scale the binned 2D values.
        scaled_values = np.interp(
            binned_values,
            (value1_min, value1_max),
            (0.0, 1.0)
        )

        # Manual brightness
        scaled_values = scaled_values * brightness

        # Expand dims for broadcasting (Y, X) -> (Y, X, 1)
        scaled_values = np.expand_dims(scaled_values, axis=2)

        # Use colours, and add scaled_values to alpha channel
        # RGB = 0.0-1.0, (Y, X, 3)
        # scaled_values = 0.0-1.0, (Y, X, 1)
        image_array = np.c_[rgb_values, scaled_values]

        image_array = np.clip(image_array, a_min=None,
                              a_max=1.0)  # TODO: Remove

        # Create the image - scale from 0.0-1.0 back to 0-255
        image_array = np.uint8(image_array * 255)
        im = Image.fromarray(image_array, 'RGBA')

        # TODO: Remove resizing
        im = im.resize(
            (image_size, image_size),
            resample=Image.NEAREST
        )  # resample=Image.BILINEAR)

        # Add mask (if supplied)
        if tile_mask is not None:
            im.putalpha(tile_mask)

        return im

    @staticmethod
    def get_circle_mask(tile_size: int, num_bins: int) -> Image:
        """
        Create mask to render bins as circles, not squares.

        Parameters
        ----------
        tile_size : int
            Square tile size in pixels (power of 2).

        num_bins : int
            Number of bins (x/y) per tile (power of 2).

        Returns
        -------
        Image :
            The mask image.

        Example
        -------
        ```
        mask = get_circle_mask(256, 64)
        img = my_image
        result = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
        result.putalpha(mask)
        ```
        """

        # Init
        circle_dtr = tile_size // num_bins
        circle_radius = circle_dtr // 2
        num_circles = num_bins
        mask_size = (circle_dtr, circle_dtr)

        # Create image (tile size)
        mask = Image.new(
            mode='L',
            size=(tile_size, tile_size),
            color=0
        )

        # Draw filled circles
        mask_draw = ImageDraw.Draw(mask)
        for x in range(num_circles):
            x_offset = x * circle_dtr
            for y in range(num_circles):
                y_offset = y * circle_dtr

                mask_draw.ellipse(
                    (
                        x_offset, y_offset,
                        x_offset + circle_dtr, y_offset + circle_dtr
                    ),
                    fill=255,
                    outline=0,
                    width=1
                )

        # Return
        return mask

    @staticmethod
    def get_scaled_brightness(binned_values,
                              zoom_level: int,
                              max_power2: int = 30,
                              smooth=False):
        """
        Gets fixed log2 brightness for bins, and adjusts for zoom level.

        Brightness of bin is determined by sum of values in the bin.
        As we zoom in, the same values will be spread over more bins/tiles = dimmer,
        so values are boosted proportionately to zoom and number of bins/tiles.


        Parameters
        ----------
        binned_values : 2D ndarray
            The binned values.

        zoom_level : int
            Current zoom level.

        max_power2 : int, optional
            The maximum power of 2 value to handle. This
            determines the final scaling to 0.0-1.0, by default 12 (4,096).
            Higher values clipped.

        smooth : bool, optional
            True includes the fractional log value whereas False discards, by default False.

        Returns
        -------
        ndarray
            The brightness values, 0.0 - 1.0.
        """

        assert binned_values.shape[0] == binned_values.shape[1], \
            'binned_values should be square'
        num_bins = binned_values.shape[0]

        # Total bin value will decrease as we zoom in,
        # as essentially same points spread over more tiles/bins.
        # Boost values according to zoom.
        total_tiles = 2**(2 * zoom_level)
        total_bins = total_tiles * (num_bins**2)

        total_bins = 1

        # Scaler (power of 2 to 0.0-1.0 range)
        scaler = 1.0 / max_power2

        # Manual brightness booster
        # TODO change back
        scaler *= 4

        # Get log2 value
        # https://en.wikipedia.org/wiki/Power_of_two
        if smooth:
            return np.log2(binned_values * total_bins) * scaler
        else:
            return np.trunc(np.log2(binned_values * total_bins)) * scaler
