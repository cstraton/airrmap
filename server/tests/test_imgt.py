# Test imgt.py

# %% Imports
import builtins
import unittest
import json
import pandas as pd
from unittest.mock import mock_open, patch

import airrmap.preprocessing.imgt as imgt


# %%
class TestImgtParser(unittest.TestCase):

    def setUp(self):
        self.description = 'HM855463|IGHV1-18*03|Homo sapiens|' + \
                           'F|V-REGION|21..316|296 nt|1| | | |98 AA|' + \
                           '98+8=106| | |'
        self.description_bad = 'f1|f2|f3|f4'
        self.v_seq = 'QVQLVQSGA.EVKKPGASVKVSCKASGYTF....' + \
            'TSYGISWVRQAPGQGLEWMGWISAY..NGNTNYA' + \
            'QKLQ.GRVTMTTDTSTSTAYMELRSLRSDDTAVYYCAR'

    def tearDown(self):
        pass

    def test_parse_description(self):
        self.assertIsInstance(imgt._parse_description(
            self.description), dict, "should be dict")
        # check it fails if unexpected number of values
        with self.assertRaises(Exception):
            _parse_description(self.description_bad)

    def test_get_imgt_residue_names(self):
        # Insertions between 111 and 112, 111A-Q->112Q-A.
        result = imgt.get_imgt_residue_names(as_dict=False)
        self.assertIsInstance(
            result, tuple, 'Tuple should be returned if as_dict=False.')
        self.assertEqual(result[-1], '129', "Final element should be '129'")
        # 111 as zero based
        self.assertEqual(result[111], '111A', '112th element should be 111A')
        # 145 as zero based
        self.assertEqual(result[145], '112', '146th element should be 112')

        result_d = imgt.get_imgt_residue_names(as_dict=True)
        self.assertIsInstance(
            result_d, dict, 'Dictionary should be returned if as_dict=True.')
        # 145 as zero based
        self.assertEqual(result_d['112'], 145,
                         'IMGT 112 should be position 146')

    def test_imgt_to_gapped(self):
        residue_names = {'111A': 0, '112A': 1, '112': 2}
        numbered_seq = {
            'region1': {
                '111A': 'A',
                # Miss out 112A, to ensure gap is added
                '112': 'B'
            }
        }

        gapped_seq = imgt.imgt_to_gapped(
            imgt_seq=numbered_seq,
            imgt_residue_names=residue_names,
            gap_char='-'
        )

        self.assertEqual(
            gapped_seq,
            'A-B',
            'Correct gapped sequence should be returned.'
        )

    def test_get_region_positions(self):

        # Tuple
        result_tuple = imgt.get_region_positions('h', as_dict=False)
        self.assertIsInstance(result_tuple, tuple, 'Tuple should be returned.')

        # Dict
        result_dict = imgt.get_region_positions('h', as_dict=True)
        self.assertIsInstance(
            result_dict, dict, 'Dictionary should be returned.')

        # is_rearranged=False
        result_dict = imgt.get_region_positions(
            '', as_dict=True, is_rearranged=False)
        self.assertEqual(
            result_dict['cdr3'][1],
            116,
            'is_rearranged=False should return 116 for cdr3 -to- position.'
        )
        self.assertTrue(
            'fw4' not in result_dict,
            'is_rearranged=False should not return fw4.'
        )

        # is_rearranged=True
        result_dict = imgt.get_region_positions(
            '', as_dict=True, is_rearranged=True)
        self.assertEqual(
            result_dict['cdr3'][1],  # 1=to position
            117,
            'is_rearranged=True should return 117 for cdr3 -to- position.'
        )
        self.assertTrue(
            'fw4' in result_dict,
            'is_arranged=True should contain fw4.'
        )

        # for_gapped_seq
        # Full length gapped seqs should be 163 chars, see imgt.get_imgt_residue_names().
        result_dict = imgt.get_region_positions(
            '', as_dict=True, is_rearranged=True, for_gapped_seq=True)
        self.assertEqual(
            result_dict['fw4'][1],  # 1=to position
            162,  # 163-1 as zero based.
            'for_gapped_seq=True should return 162 for fw4 -to- position.'
        )

    def test_parse_v_germline_to_imgt(self):
        seq_annotated = imgt._parse_v_germline_to_imgt(
            self.v_seq, 'h', remove_gaps=False)
        self.assertEqual(seq_annotated['fwh1']['1'], 'Q')
        self.assertEqual(seq_annotated['cdrh3']['106'], 'R')
        self.assertEqual(seq_annotated['fwh1']['10'], '.')

    def test_v_germline_to_imgt_remove_gaps(self):
        seq_annotated = imgt._parse_v_germline_to_imgt(
            self.v_seq, 'h', remove_gaps=True)
        seq_annotated = list(str(seq_annotated))
        self.assertNotIn('.', seq_annotated, 'Should have no gaps')

    def test_parse_json_file(self):

        # Create test records
        record1 = dict(
            name='Anchor1',
            aa_annotated=dict(
                fwh1={
                    "1": "M",
                    "2": "L"
                }
            )
        )
        record2 = dict(
            name='Anchor2',
            aa_annotated=dict(
                cdrh3={
                    "1": "G",
                }
            )
        )
        record3_no_name = dict(
            name_misspelt='Anchor3',
            aa_annotated=dict()
        )
        record_3_no_annotated = dict(
            name='Anchor3',
            aa_annotated_misspelt=dict()
        )

        records_json = '\n'.join([json.dumps(record1), json.dumps(record2)])

        # Run the function with test data
        with patch('builtins.open', mock_open(read_data=records_json)) as m:
            df: pd.DataFrame = imgt.parse_json_file(
                '/dev/null')  # any file name

        # Check we got the same back
        self.assertIsInstance(
            df, pd.DataFrame, 'DataFrame should be returned.')
        self.assertEqual(df.index.name, 'index')
        self.assertEqual(len(df.index), 2, 'DataFrame should have two rows.')
        self.assertEqual(df.iloc[0]['name'], 'Anchor1',
                         'First record should have name==Anchor1')
        self.assertDictEqual(df.iloc[0]['aa_annotated'], record1['aa_annotated'],
                             'Record1 returned annotated sequence should match test dictionary')

        # Raises error if no 'name' property
        records_json_err1 = json.dumps(record3_no_name)
        with patch('builtins.open', mock_open(read_data=records_json_err1)) as m:
            self.assertRaises(ValueError, imgt.parse_json_file, '/dev/null')


# %% Main
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
