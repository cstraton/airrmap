# %%
import unittest
import numpy as np
from airrmap.application.seqlogo import *

# %%
class TestReporting(unittest.TestCase):

      def test_get_logo(self):

        # Test data
        gapped_seqs = np.array(['A.C', 'C.C'])
        weights = np.ones(gapped_seqs.shape[0])

        # Check logo instance returned
        logo, df_counts = get_logo(
            gapped_seqs=gapped_seqs,
            weights=weights,
            title='Test Sequence',
            encode_base64=False
        )

        self.assertIsInstance(
            logo,
            logomaker.Logo,
            'logo instance should be returned.')

        self.assertEqual(
            df_counts.values.sum(),
            gapped_seqs.shape[0] * len(gapped_seqs[0]), # Use length of first seq, should all be same.
            'Sum of counts should equal number of sequences * sequence length.'
        )

        # Check base64 encoding return
        logo, df_counts = get_logo(
            gapped_seqs=gapped_seqs,
            weights=weights,
            encode_base64=True
        )
        self.assertIsInstance(
            logo, str, 'Base64 image string should be returned.')


# %%
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
