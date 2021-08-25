# %% Test OASAdapterBase (partial, WIP)

import pandas as pd
import construct
import unittest

import airrmap.preprocessing.distance as distance
from collections import OrderedDict
from airrmap.preprocessing.oas_adapter_base import OASAdapterBase, AnchorItem



class TestOASAdapterBase(unittest.TestCase):

    def setUp(self):  # Note uppercase U
        self.num_items = 329
        pass

    def tearDown(self):
        pass

    def test_anchor_dist_codec(self):
        codec = OASAdapterBase.anchor_dist_codec(self.num_items)
        self.assertIsInstance(codec, construct.Struct)

    def test_anchor_dist_encode(self):
        anchor_ids = tuple(range(self.num_items))
        anchor_dists = tuple(range(self.num_items))
        anchor_dists_d = OrderedDict(
            {k: v for k, v in zip(anchor_ids, anchor_dists)})

        codec = OASAdapterBase.anchor_dist_codec(self.num_items)
        result = OASAdapterBase.anchor_dist_encode(anchor_dists_d,
                                                   codec, 'zlib', 9)
        self.assertIsInstance(result, bytes)

    def test_anchor_dist_decode(self):
        anchor_ids = tuple(range(self.num_items))
        anchor_dists = tuple(range(self.num_items))
        anchor_dists_d = OrderedDict(
            {k: v for k, v in zip(anchor_ids, anchor_dists)})
        codec = OASAdapterBase.anchor_dist_codec(self.num_items)
        encoded = OASAdapterBase.anchor_dist_encode(anchor_dists_d,
                                                    codec, 'zlib', 9)
        decoded = OASAdapterBase.anchor_dist_decode(encoded, codec, 'zlib')

        self.assertIsInstance(decoded, construct.Container)
        anchor_dists = decoded['anchor_dists']
        self.assertEqual(decoded['dist_checksum'], int(sum(anchor_dists)),
                         'int(sum(anchor_dists)) should equal dist_checksum.')

    def test_anchor_dist_decode_num_items_mismatch(self):
        anchor_ids = tuple(range(self.num_items))
        anchor_dists = tuple(range(self.num_items))
        anchor_dists_d = OrderedDict(
            {k: v for k, v in zip(anchor_ids, anchor_dists)})
        encoder = OASAdapterBase.anchor_dist_codec(self.num_items)
        decoder = OASAdapterBase.anchor_dist_codec(self.num_items + 888)
        encoded_bytes = OASAdapterBase.anchor_dist_encode(
            anchor_dists_d, encoder)
        self.assertRaises(construct.core.StreamError, OASAdapterBase.anchor_dist_decode,
                          encoded_bytes, decoder)

    def test_get_distances(self):
        item1 = 'ABC'
        items = {
            1: AnchorItem(1, 'ABC', 0, 0),
            2: AnchorItem(2, 'AB', 0, 0)
        }

        distance_func = distance.measure_distance_lev1
        distances_d = OASAdapterBase.get_distances(item1, items, distance_func)
        self.assertDictEqual(distances_d, {1: 0, 2: 1})

    def test_compute_sequence_coords(self):
        """Compute xy for single sequence"""

        # init
        anchor_dists = OrderedDict(
            {1: 3, 2: 2, 3: 1}
        )

        anchors = {
            1: AnchorItem(anchor_id=1, seq='ABC', x=1, y=3),
            2: AnchorItem(anchor_id=2, seq='DEF', x=2, y=3),
            3: AnchorItem(anchor_id=3, seq='GHI', x=4, y=5)
        }

        num_closest_anchors = 2

        # Run
        record = OASAdapterBase.compute_sequence_coords(
            anchor_dists=anchor_dists,
            anchors=anchors,
            num_closest_anchors=num_closest_anchors
        )

        # Check
        self.assertIsInstance(record['x'], float, 'x should be numeric.')
        self.assertIsInstance(record['y'], float, 'y should be numeric.')

        self.assertEqual(record['num_closest_anchors'], num_closest_anchors,
                         'Correct number of closest anchors should be returned.')

        self.assertIsInstance(
            record['init_x'], float, 'init_x should be numeric.')
        self.assertIsInstance(
            record['init_y'], float, 'init_y should be numeric.')

        # %%


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
