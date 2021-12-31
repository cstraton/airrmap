import unittest
import itertools
from airrmap.application.reporting import *


class TestReporting(unittest.TestCase):

    def test_squeeze_gapped_seqs(self):

        seq_gapped_list1 = [
            ['AAA....AA', 1],
            ['AA......A', 1]
        ]

        self.assertListEqual(
            squeeze_gapped_seqs(seq_gapped_list1, min_gap_chars=2),
            [
                ('AAA..AA', 1),
                ('AA....A', 1)
            ],
            'Min gaps should be 2.'
        )

        self.assertListEqual(
            squeeze_gapped_seqs(seq_gapped_list1, min_gap_chars=3),
            [
                ('AAA...AA', 1),
                ('AA.....A', 1)
            ],
            'Min gaps should be 3.'
        )

        seq_gapped_list2 = [
            ['AAAAA', 1],
            ['AA..A', 1]
        ]
        self.assertListEqual(
            squeeze_gapped_seqs(seq_gapped_list2, min_gap_chars=0),
            seq_gapped_list2,
            'List should be unchanged (min is no gaps)'
        )

    def test_get_grouped_data(self):
        # Test with cdr3 lengths.

        # Generate list of records (facet row/col and length)
        rows = ('row1', 'row2')
        cols = ('col1', 'col2')
        cdr3_lengths = (1, 2, 3)
        redundancy_values = (4, 5, 6)
        records = [dict(
            facet_row=x[0],
            facet_col=x[1],
            seq_cdr3_length=x[2],
            seq_redundancy=x[3])
            for x in list(
                itertools.product(
                    rows, cols, cdr3_lengths, redundancy_values
                )
        )]
        df_records = pd.DataFrame.from_records(records)

        # Get cdr length pct distribution per facet
        # Returned as dictionary with lists for each key/field.
        result: models.ReportItem = get_grouped_data(
            df_records=df_records,
            group_field='seq_cdr3_length',
            measure_field='seq_redundancy',
            name='cdr3lengthname',
            title='CDR3 Length',
            report_type='cdr3length',
            x_label='CDR3 Length Label',
            y_label='%',
            facet_row_value='',
            facet_col_value='',
            aggregate_method='sum'
        )

        # Check keys
        self.assertEqual(result.name, 'cdr3lengthname',
                         "Report name should be 'cdr3lengthname'")
        self.assertEqual(result.title, 'CDR3 Length',
                         "Report title should be 'CDR3 Length'")
        self.assertEqual(result.report_type, 'cdr3length',
                         "Report type should be 'cdr3length'")
        self.assertEqual(result.x_label, 'CDR3 Length Label',
                         "X label should be 'CDR3 Length Label''")
        self.assertEqual(result.y_label, '%',
                         "Y label should be '%'")
        self.assertIn('data', result.__dict__,
                      "Report should contain data property")

        # Get the facet data
        result_facet_data = result.data

        # Check sum of redundancy
        total_redundancy_orig = sum([x['seq_redundancy'] for x in records])
        total_redundancy_result = sum(
            [sum(facet_data['measure_values']) for facet_data in result_facet_data])
        self.assertEqual(
            total_redundancy_orig, total_redundancy_result,
            'Total redundancy (measure_values) in result should match total redundancy in original data.'
        )

        # Check sum of measure_pcts
        # Should be 100% for each facet, total=num_facets * 100.
        num_facets = len(rows) * len(cols)
        sum_measure_pcts = sum(
            [sum(facet_data['measure_pcts']) for facet_data in result_facet_data])
        self.assertEqual(
            round(sum_measure_pcts, 2),
            num_facets * 100,
            'Measure pcts total should match 100*num_facets.'
        )

        # Check first item has the relevant columns
        # NOTE! If changing, update roi_report in front-end.
        item = result_facet_data[0]
        self.assertIsInstance(
            item['facet_row_value'],
            str,
            "Result should contain 'facet_row_value' with a 'str' instance."
        )
        self.assertIsInstance(
            item['facet_col_value'],
            str,
            "Result should contain 'facet_col_value' with a 'str' instance."
        )
        self.assertIsInstance(
            item['group_values'],
            list,
            "Result should contain 'group_values' with a 'list' instance."
        )
        self.assertIsInstance(
            item['measure_values'],
            list,
            "Result should contain 'measure_values' with a 'list' instance."
        )
        self.assertIsInstance(
            item['measure_pcts'],
            list,
            "Result should contain 'measure_pcts' with a 'list' instance."
        )

    def test_get_report_info(self):
        # init

        # Generate list of records (facet row/col and length)
        rows = ('row1', 'row2')
        cols = ('col1', 'col2')
        records = [dict(
            facet_row=x[0],
            facet_col=x[1])
            for x in list(
                itertools.product(
                    rows, cols
                )
        )]
        df_records = pd.DataFrame.from_records(records)

        # Get report information
        result: models.ReportItem = get_report_info(df_records)

        # Check keys
        self.assertEqual(result.report_type, 'reportinfo',
                         "Report type should be 'reportinfo'")
        self.assertIsInstance(
            result.data, dict, "Report data property should be 'dict' type")

        # Check record_count
        self.assertEqual(result.data['record_count'], len(
            df_records.index), 'Record count should return correct number of records')

    def test_build_seq_logo(self):

        # Generate test records
        rec1 = {
            f"seq_gapped_cdr1": 'AAA...AA',
            "redundancy": 2
        }
        rec2 = {
            f"seq_gapped_cdr1": 'AA.....A',
            "redundancy": 4
        }
        df_records = pd.DataFrame.from_records([rec1, rec2])

        # Get the logo and metadata
        result: Dict = build_seq_logo(
            title='CDR1',
            gapped_field = 'seq_gapped_cdr1',
            df_records=df_records,
            redundancy_field='redundancy'
        )

        # Expected keys
        expected_keys = (
            'seqs_top',
            'seqs_bottom',
            'seqs_unique_count',
            'region_name',
            'logo_img'
        )

        # Check dictionary returned
        self.assertTrue(
            all(k in result for k in expected_keys),
            f'Result should contain the following keys: {expected_keys}'
        )


# %%
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
