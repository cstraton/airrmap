# Rebuild the index

# %%
import airrmap.preprocessing.indexer as indexer
from airrmap.application.config import AppConfig; 


# %%
indexer.build_index(AppConfig())