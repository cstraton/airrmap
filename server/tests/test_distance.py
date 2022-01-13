
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

    def test_euclidean(self):
        """Test euclidean"""
        self.assertEqual(
            compute_euclidean(
                x1=0,
                y1=0,
                x2=5,
                y2=7
            ),
            ((5**2) + (7**2))**0.5,
            'Correct distance should be returned (Pythagorean Theorem).'
        )

    def test_measure_distance1(self):
        """Check correct distance reported for different lengths"""

        # Create tests
        # 0: String 1
        # 1: String 2
        # 2: Expected return value
        # 3: Message
        string_values = [
            ('Hello', 'Hello', 0, 'Should be 0'),
            ('Hello1', 'Hello', 1, 'Should be 1'),
            ('Hello', 'Hello1', 1, 'Should be 1'),
            ('abcd', 'dcba', 4, 'Should be 4')
        ]
        env_kwargs = dict()
        record_kwargs = dict(seq='seq')

        for i in range(len(string_values)):
            self.assertEqual(
                measure_distance1(
                    record1=dict(seq=string_values[i][0]),
                    record2=dict(seq=string_values[i][1]),
                    record1_kwargs=record_kwargs,
                    record2_kwargs=record_kwargs,
                    env_kwargs=env_kwargs
                ),
                string_values[i][2],
                string_values[i][3]
            )

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
        item1_trailing_spaces = dict(
            dict(
                numbered_seq_field=dict(
                    cdr1={'i1 ': 'A', 'i2 ': 'B'},
                    cdr2={'i1 ': 'C', 'i2 ': 'D'})
            )
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

        item1_json_single_quotes = dict(
            numbered_seq_field=item1_json['numbered_seq_field'].replace(
                '"', "'")
        )
        item3_json_single_quotes = dict(
            numbered_seq_field=item3_json['numbered_seq_field'].replace(
                '"', "'")
        )

        # Set up kwargs
        record_kwargs = dict(
            convert_json_single_quoted=False,
            numbered_seq_field='numbered_seq_field'
        )
        record_kwargs_single_quoted = record_kwargs.copy()
        record_kwargs_single_quoted['convert_json_single_quoted'] = True

        env_kwargs = dict(
            regions=['cdr1', 'cdr2']
        )

        # Same
        self.assertEqual(
            measure_distance3(
                record1=item1,
                record2=item1,
                record1_kwargs=record_kwargs,
                record2_kwargs=record_kwargs,
                env_kwargs=env_kwargs
            ), 0, 'Same')

        # Same, but with trailing spaces in residue numbers
        self.assertEqual(
            measure_distance3(
                record1=item1_trailing_spaces,
                record2=item1,
                record1_kwargs=record_kwargs,
                record2_kwargs=record_kwargs,
                env_kwargs=env_kwargs
            ), 0, 'Same with trailing spaces in residue keys')

        # Same number, different values
        self.assertEqual(
            measure_distance3(
                record1=item1,
                record2=item3,
                record1_kwargs=record_kwargs,
                record2_kwargs=record_kwargs,
                env_kwargs=env_kwargs
            ), 4, 'Different values')

        # Same number, different values, double-quoted JSON string (check conversion to dict)
        self.assertEqual(
            measure_distance3(
                record1=item1_json,
                record2=item3_json,
                record1_kwargs=record_kwargs,
                record2_kwargs=record_kwargs,
                env_kwargs=env_kwargs
            ), 4, 'Different values (double-quoted json)')

        # Same number, different values, single-quoted JSON string
        self.assertEqual(
            measure_distance3(
                record1=item1_json_single_quotes,
                record2=item3_json_single_quotes,
                record1_kwargs=record_kwargs_single_quoted,
                record2_kwargs=record_kwargs_single_quoted,
                env_kwargs=env_kwargs
            ), 4, 'Different values (single-quoted json)')

        # +1 Extra
        self.assertEqual(
            measure_distance3(
                record1=item1,
                record2=item2,
                record1_kwargs=record_kwargs,
                record2_kwargs=record_kwargs,
                env_kwargs=env_kwargs
            ), 1, 'Extra')

        # cdr2 only
        self.assertEqual(
            measure_distance3(
                record1=item1,
                record2=item2,
                record1_kwargs=record_kwargs,
                record2_kwargs=record_kwargs,
                env_kwargs=dict(regions=['cdr2'])
            ), 0, 'CDR2 only')

        # Exception if region doesn't exist
        self.assertRaises(KeyError,
                          measure_distance3,
                          item1,
                          item1,
                          record_kwargs,
                          record_kwargs,
                          dict(regions=['bad_region']))

        # Exception if no regions
        self.assertRaises(Exception,
                          measure_distance3,
                          item1,
                          item1,
                          record_kwargs,
                          record_kwargs,
                          dict(regions=[]))

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
        record_kwargs = dict(
            columns=['col1', 'col2']
        )
        env_kwargs = None

        self.assertEqual(
            measure_distance_lev2(
                record1=rec1,
                record2=rec2,
                record1_kwargs=record_kwargs,
                record2_kwargs=record_kwargs,
                env_kwargs=env_kwargs
            ),
            3, # Number of characters different
            'Correct distance should be returned.'
        )

        self.assertEqual(
            measure_distance_lev2(
                record1=dict(
                    col1="ABC",
                    col2="DEF"
                ),
                record2=dict(
                    col1=pd._libs.missing.NAType(), # NA Type
                    col2="DEF"  # make same as record1, col2
                ),
                record1_kwargs=record_kwargs,
                record2_kwargs=record_kwargs,
                env_kwargs=env_kwargs
            ),
            3, # Should be 3 characters different (record1,col1 "ABC" vs record2,col1=NA/'')
            'Should treat NA types as zero length string (no error).'
        )

    def test_create_distance_matrix(self):
        """Check distance matrix"""
        records = self.records
        distance_func = measure_distance1
        record_kwargs = dict(
            seq='seq'
        )
        df = create_distance_matrix(
            records=records,
            distance_function=distance_func,
            record_kwargs=record_kwargs,
            env_kwargs=dict()
        )
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
        record_kwargs = dict(
            seq='seq'
        )
        distmatrix = create_distance_matrix(
            records=records,
            distance_function=distance_func,
            record_kwargs=record_kwargs,
            env_kwargs=dict()
        )

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
