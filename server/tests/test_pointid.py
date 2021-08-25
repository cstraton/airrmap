# Tests for pointid.py

# %% Imports
import unittest
from airrmap.preprocessing.pointid import *

# %% Tests
class TestPointID(unittest.TestCase):

    def test_create_point_id(self):
        file_id = int(0xFFFF) # 16 bit
        seq_id = int(0xFFFFFFFFFFFF) # 48 bit
        expected = int(0xFFFFFFFFFFFFFFFF) # 64 bit
        result = create_point_id(file_id, seq_id)

        # Check expected value returned
        self.assertEqual(result, expected, 'Max 64 bit value should be returned.')

        # If max exceeded
        with self.assertRaises(Exception):
            create_point_id(file_id+1, seq_id)
        with self.assertRaises(Exception):
            create_point_id(file_id, seq_id+1)

        # If negative
        with self.assertRaises(Exception):
            create_point_id(-1, seq_id)
        with self.assertRaises(Exception):
            create_point_id(file_id, -1)

    def test_invert_point_id(self):
        point_id = int(0xFFFFFFFFFFFFFFFF) # 64 bit
        file_id, seq_id = invert_point_id(point_id)

        # Check expected value returned
        self.assertEqual(file_id, 0xFFFF, 'Expected file_id should be returned.')
        self.assertEqual(seq_id, 0xFFFFFFFFFFFF, 'Expected seq_id should be returned.')

       


# %% Main
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
