# Example script to apply custom transformations applied to CSV data units
# prior to running the pre-processing routine.

# Determines folder locations and parameters using
# the environment configuration file.

# To Use:
# 1. Ensure envconfig.yaml file has been set up.
# 2. Place sequence files in an 'original' subfolder (e.g. /src/seq/original/)
# 3. SKIPS existing processed files. Ensure sequence folder is empty (except for 'original' subfolder). 
# 4. Take a copy of this script, and save in the env folder.
# 5. Set the env_name in Init()
# 6. Write the code for the transform
# 7. Run - files will be saved to the /src/seq/ (or folder specified in envconfig.yaml)

# %% Imports
import pandas as pd
import os
import glob
import gzip
from typing import Callable
from pathlib import Path
from tqdm import tqdm
from airrmap.application.config import AppConfig, SeqFileType

# %% Transformation goes here:
def transform(df: pd.DataFrame):
    """
    Custom transformation code

    Parameters
    ----------
    df : pd.DataFrame
        The original dataframe

    Returns
    -------
    pd.DataFrame
        The transformed dataframe
    """
    return df[df['ANARCI_status'] == 'good']


# %% Init
env_name = 'Schultheiss_CDRH1_CDRH2'
stop_after_n_files = 0 # For testing, 0 to process all rows.
appcfg = AppConfig()
envcfg = appcfg.get_env_config(env_name)
env_folder = appcfg.get_env_folder(env_name)
seqcfg = envcfg['sequence']
seq_folder = os.path.join(env_folder, seqcfg['src_folder'])
seq_orig_folder = os.path.join(seq_folder, 'original')
seq_pattern = seqcfg['src_file_pattern']
seq_row_start = seqcfg['seq_row_start']
seq_skip_rows = seq_row_start - 2 # For Pandas, -1 for csv headers and rest for other rows (e.g. file header)

# %% Get list of sequence files matching the pattern (from -original- folder)
seq_files = glob.glob(os.path.join(seq_orig_folder, seq_pattern))

# %% Init progress
file_ctr = 0
pbar = tqdm(total=len(seq_files), desc='Processing files...', position=0, leave=True)

# %% Start processing files
for seq_fn in seq_files:

    # Update ctr
    file_ctr += 1
    pbar.update(1)
    seq_name = Path(seq_fn).name
    pbar.set_description(desc=f'Processing {seq_name}...')
    seq_fn_out = os.path.join(seq_folder, seq_name)

    # Skip if file already exists
    if os.path.isfile(seq_fn_out):
        print(f'SKIPPING {seq_name} as already exists...')
        continue

    # %% Open the file
    file_header_lines = []
    with gzip.open(seq_fn, 'rt') as f:

        # Keep header lines (and line returns)
        for i in range(seq_skip_rows):
            file_header_lines.append(str(next(f)))

        # Read rest into DataFrame
        df: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=f
        )

        # Transform
        df = transform(df)

        # %% Write out
        with gzip.open(seq_fn_out, 'wt') as f_out:
            f_out.write(''.join(file_header_lines)) # Will already have '\n from next()
            df.to_csv(f_out, header=True, index=False)

    # Stop if required
    if file_ctr >= stop_after_n_files and stop_after_n_files>0:
        print(f'Stopping after {stop_after_n_files} file(s).')
        break

# Show confirmation
pbar.close()
print (f'Finished, processing {file_ctr} file(s).')