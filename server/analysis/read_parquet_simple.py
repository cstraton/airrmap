# Quickly read a single parquet file (or all files)

# %% Imports
import pandas as pd
from airrmap.application.config import AppConfig, SeqFileType

# %% Set params
env_name = 'ENV_NAME'
file_type = SeqFileType.RECORD

# %% Get files
appcfg = AppConfig()
seq_files = appcfg.get_seq_files(env_name, file_type)

# %% Read
df = pd.read_parquet(seq_files[0])
df.head(10)


# %%
