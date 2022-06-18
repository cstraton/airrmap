# Starter template for performing analysis on the data

# %% Imports
import pandas as pd
import airrmap.shared.analysis_helper as ah
from airrmap.shared.models import RepoQuery

# %% Init
repo, cfg = ah.get_repo_cfg()

# %% Get query
query = RepoQuery('ENV_NAME',
                  value1_field='redundancy',
                  value2_field='cdr3',
                  facet_col='f.Longitudinal',
                  facet_row='f.Subject'
                  )

# %% Get data
df, report = repo.run_coords_query(cfg, query.to_dict())
