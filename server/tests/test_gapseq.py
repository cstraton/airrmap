# %% Imports
import unittest
from airrmap.preprocessing.gapseq import *

# %%


class TestSeq(unittest.TestCase):

    def test_get_gapped_seq(self):

        self.assertEquals(
            get_gapped_seq(
                seq='123',
                fixed_length=3
            ),
            '123',
            'No gaps should be inserted if fixed_length = seq length.'
        )

        self.assertEquals(
            get_gapped_seq(
                seq='1234',
                fixed_length=6,
                gap_char='.'
            ),
            '12..34',
            'Even-length sequences should be handled correctly.'
        )

        self.assertEquals(
            get_gapped_seq(
                seq='12345',
                fixed_length=6,
                gap_char='.',
                left_bias=True
            ),
            '123.45',
            'Odd-length sequence with left bias should be handled correctly.'
        )

        self.assertEquals(
            get_gapped_seq(
                seq='12345',
                fixed_length=6,
                gap_char='.',
                left_bias=False
            ),
            '12.345',
            'Odd-length sequence with right bias should be handled correctly.'
        )

        self.assertEquals(
            get_gapped_seq(
                seq='12345',
                fixed_length=12,
                gap_char=':',
                left_bias=False
            ),
            '12:::::::345',
            'Longer fixed length with different gap character should be handled correctly.'
        )

        with self.assertRaises(ValueError):
            # Should raise error if fixed length is less than seq length
            get_gapped_seq(
                seq='123',
                fixed_length=2
            )


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
