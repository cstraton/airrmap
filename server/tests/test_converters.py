# Test converters.py

# %% Imports
import unittest
from airrmap.application.converters import *

# %% Tests
class TestConverters(unittest.TestCase):

    def test_latlng_to_xy(self):
        latlngs = [dict(lat=1, lng=5),
                   dict(lat=2, lng=6)]

        xys = latlng_to_xy(latlngs)
        self.assertListEqual(
            xys,
            [[5, 1], [6, 2]],
            'Correct conversion from LatLng to XY list should be returned'
        )


# %% Main
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
