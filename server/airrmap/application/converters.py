"""
Conversion functions
"""

# Unit Tests
# tests > test_converters.py

# %% imports
from typing import Sequence, Dict, Tuple


# %%
def latlng_to_xy(latlngs: Sequence[Dict]):
    return [[item['lng'], item['lat']] for item in latlngs]
