
import unittest

from airrmap.preprocessing.distance import *

# %% Unit tests


class TestDistances(unittest.TestCase):

    def setUp(self):
        records_d = {'rec1': {'seq': 'ABC'}, 'rec2': {'seq': 'XBC'}}
        self.records = pd.DataFrame.from_dict(records_d, orient="index")
        pass

    def tearDown(self):
        pass

    def test_measure_distance1(self):
        """Check correct distance reported for different lengths"""
        self.assertEqual(measure_distance1('Hello', 'Hello'), 0, "Should be 0")
        self.assertEqual(measure_distance1(
            'Hello1', 'Hello'), 1, "Should be 1")
        self.assertEqual(measure_distance1(
            'Hello', 'Hello1'), 1, "Should be 1")
        self.assertEqual(measure_distance1('abcd', 'dcba'), 4, "Should be 4")

    def test_measure_distance2(self):
        self.assertEqual(measure_distance2('Hello', 'Hello'), 0, "Should be 0")
        self.assertEqual(measure_distance2(
            'Hello1', 'Hello'), 1, "Should be 1")
        self.assertEqual(measure_distance2(
            'Hello', 'Hello1'), 1, "Should be 1")
        self.assertEqual(measure_distance2('abcd', 'dcba'), 2, "Should be 2")
        self.assertAlmostEqual(measure_distance2(
            'abcdef', 'edcba'), 2.2361, places=1)

    def test_measure_distance3(self):
        item1 = dict(
            numbered_seq_field=dict(
                cdr1=dict(i1='A', i2='B'),
                cdr2=dict(i1='C', i2='D'))
        )
        item2 = dict(
            numbered_seq_field=dict(
                cdr1=dict(i1='A', i2='B', i3='Extra'),
                cdr2=dict(i1='C', i2='D'))
        )

        item3 = dict(
            numbered_seq_field=dict(
                cdr1=dict(i1='U', i2='V'),
                cdr2=dict(i1='Y', i2='Z'))
        )

        item1_json = dict(
            numbered_seq_field=json.dumps(item1['numbered_seq_field'])
        )
        item3_json = dict(
            numbered_seq_field=json.dumps(item3['numbered_seq_field'])
        )

        # Same
        self.assertEqual(
            measure_distance3(
                record1=item1,
                record2=item1,
                numbered_seq_field='numbered_seq_field',
                regions=['cdr1', 'cdr2']
            ), 0, 'Same')

        # Same number, different values
        self.assertEqual(
            measure_distance3(
                record1=item1,
                record2=item3,
                numbered_seq_field='numbered_seq_field',
                regions=['cdr1', 'cdr2']
            ), 4, 'Different values')

        # Same number, different values, JSON string (check conversion to dict)
        self.assertEqual(
            measure_distance3(
                record1=item1_json,
                record2=item3_json,
                numbered_seq_field='numbered_seq_field',
                regions=['cdr1', 'cdr2']
            ), 4, 'Different values (json)')

        # +1 Extra
        self.assertEqual(
            measure_distance3(
                record1=item1,
                record2=item2,
                numbered_seq_field='numbered_seq_field',
                regions=['cdr1', 'cdr2']
            ), 1, 'Extra')

        # cdr2 only
        self.assertEqual(
            measure_distance3(
                record1=item1,
                record2=item2,
                numbered_seq_field='numbered_seq_field',
                regions=['cdr2']
            ), 0, 'CDR2 only')

        # Exception if region doesn't exist
        self.assertRaises(KeyError,
                          measure_distance3,
                          item1,
                          item1,
                          'numbered_seq_field',
                          ['bad_region'])

        # Exception if no regions
        self.assertRaises(Exception,
                          measure_distance3,
                          item1,
                          item1,
                          'numbered_seq_field',
                          [])

    def test_measure_distance_lev1(self):
        self.assertEqual(
            measure_distance_lev1("ABC", "ABC"),
            0,
            'Same string, should be distance 0.'
        )

        self.assertEqual(
            measure_distance_lev1("ABC", "ABD"),
            1,
            '1 character difference, should be distance 1.'
        )

        self.assertEqual(
            measure_distance_lev1("ABC", "ABCD"),
            1,
            '1 character longer, should be distance 1.'
        )

        self.assertEqual(
            measure_distance_lev1("ABC", "CBA"),
            2,
            'Reversed, should be distance 2.'
        )

        self.assertEqual(
            measure_distance_lev1("ABC", "A-B-C"),
            2,
            'Inserts, should be distance 2.'
        )

    def test_measure_distance_lev2(self):
        # Levenshtein distance with
        # support for concatenation of multiple columns.
        # Create two records, each with two columns.
        rec1 = dict(
            col1="ABC",
            col2="DEF"
        )
        rec2 = dict(
            col1="ABD",  # dist=1
            col2="DGH"  # dist=2
        )
        kwargs = dict(
            columns=['col1', 'col2']
        )

        self.assertEqual(
            measure_distance_lev2(
                rec1,
                rec2,
                **kwargs
            ),
            3,
            'Distance should be 3.'
        )

    def test_create_distance_matrix(self):
        """Check distance matrix"""
        records = self.records
        distance_func = measure_distance1
        measure_value = 'seq'
        df = create_distance_matrix(records, distance_func, measure_value)
        self.assertEqual(df['rec1']['rec2'], 1, "Should be 1")
        self.assertEqual(df['rec2']['rec1'], 1, "Should be 1")
        self.assertEqual(df['rec1']['rec1'], 0, "Should be 0")
        self.assertEqual(df['rec2']['rec2'], 0, "Should be 0")

    def test_xy(self):
        # Tests functions related to xy coordinates:
        # compute_xy()
        # compute_xy_distmatrix()
        # compute_delta()
        # melt_distmatrices()

        # create_distance_matrix()
        records = self.records
        distance_func = measure_distance1
        measure_value = 'seq'
        distmatrix = create_distance_matrix(
            records, distance_func, measure_value)

        # compute_xy()
        xycoords = compute_xy(distmatrix, 'MDS', random_state=2)
        self.assertEqual(
            len(xycoords.index),
            len(records.index), 'compute_xy (MDS)')
        self.assertEqual(
            len(compute_xy(distmatrix, 'TSNE', random_state=2).index),
            len(records.index), 'compute_xy (t-SNE)')
        self.assertEqual(
            len(compute_xy(distmatrix, 'PCA', random_state=2).index),
            len(records.index), 'compute_xy (PCA)')

        # compute_xy_distmatrix()
        xy_distmatrix = compute_xy_distmatrix(xycoords, distmatrix)
        self.assertEqual(xy_distmatrix.shape, distmatrix.shape,
                         'compute_xy_distmatrix()')

        # compute_delta()
        delta_matrix = compute_delta(distmatrix, xy_distmatrix)
        self.assertEqual(delta_matrix.shape,
                         distmatrix.shape, 'compute_delta()')

        # melt_distmatrices()
        melted_dists = melt_distmatrices(
            distmatrix, xy_distmatrix, delta_matrix)
        self.assertEqual(len(melted_dists.index),
                         distmatrix.shape[0] * distmatrix.shape[1],
                         'melt_distmatrices()')

    def test_verify_triangle_inequality(self):
        # Distance matrix
        # A and B are arbitrary lengths
        # C calculated using C^2 = A^2 + B^2
        # D is zero (to break triangle inequality)
        records = records = {'A': {'A': 0, 'B': 2, 'C': 3, 'D': 0},
                             'B': {'A': 2, 'B': 0, 'C': 3.6, 'D': 0},
                             'C': {'A': 3, 'B': 3.6, 'C': 0, 'D': 0},
                             'D': {'A': 0, 'B': 0, 'C': 0, 'D': 0}
                             }
        records = pd.DataFrame.from_dict(records, orient='index')
        results = verify_triangle_inequality(records, 100)

        # All results should have 'D' in (which breaks triangle inequality)
        results_issue = [x for x in results if 'D' not in x]

        self.assertGreater(len(results), 0, 'Some results should be returned')
        self.assertEqual(len(results_issue), 0,
                         'Only "D" should break triangle inequality')


# %% Main
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=1, exit=False)
