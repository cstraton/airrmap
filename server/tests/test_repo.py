# Import
import itertools
import unittest
import random
import pandas as pd

from airrmap.application.repo import DataRepo


class TestRepo(unittest.TestCase):

    @staticmethod
    def get_dummy_data(facet_rows=3,
                       facet_cols=3,
                       data_rows_per_facet=2) -> pd.DataFrame:
        """Generate some dummy data for testing"""
        facet_row_values = [f'row{i+1}' for i in range(facet_rows)]
        facet_col_values = [f'col{i+1}' for i in range(facet_cols)]
        facet_values = itertools.product(facet_row_values, facet_col_values)

        records = []
        for row_value, col_value in facet_values:
            for i in range(data_rows_per_facet):
                record = dict(
                    facet_row=row_value,
                    facet_col=col_value,
                    x=random.random(),
                    y=random.random(),
                    value1=random.randint(0, 10),
                )
                records.append(record)

        df = pd.DataFrame.from_records(records)
        return df

    def test_split_by_facet(self):
        # Init
        facet_rows = 2
        facet_cols = 2
        facet_row_column = 'facet_row'
        facet_col_column = 'facet_col'
        data_rows_per_facet = 2

        # Dummy data
        df = TestRepo.get_dummy_data(
            facet_rows=facet_rows,
            facet_cols=facet_cols,
            data_rows_per_facet=data_rows_per_facet
        )

        # Split by facet
        d_split = DataRepo.split_by_facet(
            df,
            facet_row_column=facet_row_column,
            facet_col_column=facet_col_column
        )

        # Check length is as expected
        self.assertEqual(
            len(d_split),
            facet_rows * facet_cols,
            'split_by_facet() should return one group per facet.'
        )

        # Check contents
        # Format:
        # {
        #   ('row1', 'col1'): pd.DataFrame
        #   ('row1', 'col2'): pd.DataFrame
        #   ...
        # }
        for key, df_facet in d_split.items():
            facet_row_value, facet_col_value = key
            row_value_unique = tuple(set(df_facet[facet_row_column]))
            col_value_unique = tuple(set(df_facet[facet_col_column]))

            self.assertEqual(
                len(row_value_unique),
                1,
                'Each group should contain only one facet_row value.'
            )
            self.assertEqual(
                facet_row_value,
                row_value_unique[0],
                'Each group should be set to the correct facet_row value.'
            )
            self.assertEqual(
                len(col_value_unique),
                1,
                'Each group should contain only one facet_col value.'
            )
            self.assertEqual(
                facet_col_value,
                col_value_unique[0],
                'Each group should be set to the correct facet_col value.'
            )


# %%
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
