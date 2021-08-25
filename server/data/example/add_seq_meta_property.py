# Example of adding a custom property to the Parquet sequence meta files.

# 1. Copy to environment folder.
# 2. Adjust as required.

# Add on the Disease2 file meta property
# for the Eliyah environment
# Can be run multiple times (will update instead of adding new property.)
# Remember to run runner_index.py afterwards.

# %% Imports
import pandas as pd
import os
from tqdm import tqdm
from airrmap.application.config import AppConfig, SeqFileType


def add_disease2_property(df: pd.DataFrame) -> pd.DataFrame:

    # %% Get the Subject record and values
    df_subject = df[df['property_name'] == 'Subject']
    assert len(df_subject.index) == 1, 'Expecting exactly 1 Subject record.'
    subject = str(df_subject['property_value'].iloc[0]).upper()
    new_property_d = df_subject.iloc[0].to_dict()

    # Get the value for the new property
    if subject[:2] == 'CI':
        new_value = 'CI'
    elif subject[:2] == 'SC':
        new_value = 'SC'
    else:
        new_value = 'Control'

    # %% Set the property (add or update)
    new_property = 'Disease2'
    df_new_property = df[df['property_name'] == new_property]
    if len(df_new_property.index) == 0:
        # Add the new property if it doesn't exist
        new_property_d['property_name'] = new_property
        new_property_d['property_value'] = new_value
        df = df.append(new_property_d, ignore_index=True,
                       verify_integrity=True)
    elif len(df_new_property.index) == 1:
        # Otherwise, update existing property
        row_index = df_new_property.index[0]
        df._set_value(row_index, 'property_name', new_property)
        df._set_value(row_index, 'property_value', new_value)
    else:
        raise Exception('Unexpected number of records found for new property.')

    return df


def process_file(fn: str, out_fn: str, compression: str):
    """Process a single file"""

    # Read
    df: pd.DataFrame = pd.read_parquet(fn)

    # Add on new property
    df = add_disease2_property(df)

    # Save
    df.to_parquet(out_fn, compression=compression)


# %% Init
env_name = 'Eliyahu_CDRH1_CDRH2'
appcfg = AppConfig()
envcfg = appcfg.get_env_config(env_name)
compression = envcfg['sequence']['output_file_compression']

# %% Get meta files
meta_files = appcfg.get_seq_files(
    env_name=env_name,
    file_type=SeqFileType.META
)

# %% Process
for fn in tqdm(meta_files):
    process_file(
        fn=fn,
        out_fn=fn,
        compression=compression
    )

# %% Confirm
print('Completed, remember to refresh the Metadata Index.')
