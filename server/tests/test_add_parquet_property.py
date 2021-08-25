# %% imports
import io
import unittest
import tempfile
from airrmap.util.add_parquet_property import *


class TestAddParquetProperty(unittest.TestCase):

    def test_add_parquet_property(self):
        # Test adding a new property to a Parquet file.
        # Uses test files in memory using BytesIO.

        # Init
        length_multiplier = 2

        # Create 2 test records
        records = [
            dict(field1='ab'),
            dict(field1='abc')
        ]
        df_records = pd.DataFrame.from_records(records)

        # Function to return length (multiplied)
        transform_func = lambda row, **kwargs: dict(
            field1_length=len(row['field1']) * kwargs['multiplier'])

        # Write to Parquet, read, add property and write out.
        with io.BytesIO() as f1:
            with io.BytesIO() as f2:
                # Write to parquet
                df_records.to_parquet(f1)

                # Read, add property and write result
                add_parquet_property(
                    fn_in=f1,
                    fn_out=f2,
                    transform_func=transform_func,
                    transform_kwargs=dict(multiplier=length_multiplier),
                    compression='snappy',
                    chunk_size=1024
                )

                # Read parquet to Pandas
                df_records_modified = pd.read_parquet(f2)

        # Check
        self.assertEqual(
            len(df_records_modified.index),
            2,
            'Modified file should still have 2 records.'
        )
        self.assertEqual(
            df_records_modified['field1_length'].iloc[0],
            len('ab') * length_multiplier,
            'Modified record should have new field1_length property.'
        )


# %%
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
