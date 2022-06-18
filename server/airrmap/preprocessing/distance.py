"""Distance measures between receptors"""

# tests > test_distance.py

# %% Imports
import pandas as pd
import numpy as np
import itertools
import random
import collections
import json
from numba import njit
from polyleven import levenshtein
from tqdm import tqdm
from sklearn import manifold
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from typing import Callable, Dict, Optional, Any


@njit
def compute_euclidean(x1, y1, x2, y2):
    """
    Euclidean distance.

    Calculate the Euclidean distance
    between a pair of Cartesian coordinates
    (c^2 = a^2 + b^2)
    """

    dx = (x2 - x1)**2
    dy = (y2 - y1)**2
    return (dx + dy)**0.5


def measure_distance1(record1: Dict, record2: Dict, record1_kwargs: Dict, record2_kwargs: Dict, env_kwargs: Dict):
    """
    Measure distance 1.

    Compare each character position
    starting from the first character.
    Distance is +1 for every different
    or different or additional character.

    record1_kwargs and record2_kwargs should
    contain a 'seq' property with the string
    to compare.
    """

    string1 = record1[record1_kwargs['seq']]
    string2 = record2[record2_kwargs['seq']]

    # Get longest (a) and shortest (b)
    if len(string1) >= len(string2):
        stringa = string1
        stringb = string2
    else:
        stringa = string2
        stringb = string1

    # Make arrays
    a = np.array(list(stringa))
    b = np.array(list(stringb))
    shortest_len = len(b)

    # Default to +1 difference
    adistance = np.ones(len(a))

    # Compare strings (0 for matches)
    long_short_diff = b != a[:shortest_len]
    adistance[:shortest_len] = long_short_diff[:]

    # Return
    return adistance.sum()


# %%
def measure_distance2(string1: str, string2: str):
    """
    Measure distance 2.

    Compares strings on char-by-char basis.
    Creates vector of per-char distances then computes
    the euclidean distance from that. Assumes
    strings are only truncated from the right
    (i.e. left-most chars are always comparable).

    Distance Scoring:
    0: leftmost characters match
    +1: char not matched, or additional characters in longest string

    :param string1: first string
    :param string2: second string

    :returns: euclidean distance between char distance vectors
    """

    # Get longest (a) and shortest (b)
    if len(string1) >= len(string2):
        longest_str = string1
        shortest_str = string2
    else:
        longest_str = string2
        shortest_str = string1

    alongest_str = np.array(list(longest_str))
    ashortest_str = np.array(list(shortest_str))
    longest_len = len(alongest_str)
    shortest_len = len(ashortest_str)

    # Init distance array (1 element per character)
    # Default to +1 difference
    dist_array = np.ones(longest_len)

    # Compare strings (0s if matching)
    long_short_diff = ashortest_str != alongest_str[:shortest_len]

    # Overwrite +1 differences with 0s where matching
    dist_array[:shortest_len] = long_short_diff[:]

    # We have a one-hot array now...
    # Compute the euclidean distance
    # (could use numbers other than 1 if using weights)
    # Distance derived from (we use distance, not coords though)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html
    dist = np.square(dist_array).sum() ** 0.5

    return dist


def measure_distance3(record1: Any, record2: Any, record1_kwargs: Dict, record2_kwargs: Dict, env_kwargs: Dict):
    """
    Measure distance 3.

    Note: Keys and values will be trimmed and
          comparison will be case insensitive.
    Computes the distance between two
    annotated sequences. +1 for each residue different.
    CARE!: +1 distance will be added to any gaps - consider this if using different
    lengths / partials, or if the annotated sequence contains '.' items.

    Parameters
    ----------
    record1 : Any
        Dict-like record.

    record2 : Any
        Dict-like record.

    record1_kwargs : Dict
        Should contain a 'numbered_seq_field' property
        that points to the property in record1 containing
        a Dict-like or JSON-annotated sequence.
        e.g.: record {} -> region ('cdrh2') -> residue number (imgt '26').
        Should contain 'convert_json_single_quoted' property, where True
        replaces single-quoted JSON strings, so they can be loaded
        by json.loads() (assumes no values have single quotes in).

    record2_kwargs : Dict
        Should contain a 'numbered_seq_field' property
        that points to the property in record2 containing
        a Dict-like or JSON-annotated sequence.
        e.g.: record {} -> region ('cdrh2') -> residue number (imgt '26').
        Should contain 'convert_json_single_quoted' property, where True
        replaces single-quoted JSON strings, so they can be loaded
        by json.loads() (assumes no values have single quotes in).

    env_kwargs : Dict
        Should contain a 'regions' property; a List containing
        the region names to compare.
        e.g. ['cdrh1', 'cdrh2']

    Returns
    -------
    int
        The computed distance.

    Raises
    ------
    Exception
        If either item/record doesn't contain all regions.
    """

    # Ensure at least one region
    regions = env_kwargs['regions']
    if len(regions) < 1:
        raise Exception('No regions provided')

    # Get numbered sequence field
    item1 = record1[record1_kwargs['numbered_seq_field']]
    item2 = record2[record2_kwargs['numbered_seq_field']]

    # Convert single quotes if required
    # NOTE: assumes no values with single quotes
    # ast.literal_eval() is too slow, so convert and use json.loads.
    if record1_kwargs['convert_json_single_quoted'] == True:
        item1 = item1.replace("'", '"')
    if record2_kwargs['convert_json_single_quoted'] == True:
        item2 = item2.replace("'", '"')

    # Load JSON
    if not isinstance(item1, collections.Mapping):
        item1 = json.loads(item1)
    if not isinstance(item2, collections.Mapping):
        item2 = json.loads(item2)

    # Perform distance measurement
    total_distance = 0
    for region_key in regions:
        region1 = item1[region_key]
        region2 = item2[region_key]

        # Trim and make case insensitve
        # OAS has trailing space for annotated residue keys,
        # which can cause issues during comparison to anchors.
        # (Jan 2022).
        region1 = {k.strip().upper(): v.strip().upper()
                   for k, v in region1.items()}
        region2 = {k.strip().upper(): v.strip().upper()
                   for k, v in region2.items()}

        set1 = set(region1.keys())
        set2 = set(region2.keys())

        # Count differences where a residue is in
        # both sets
        setintersect = set1.intersection(set2)
        uniondiff = 0
        for k in setintersect:
            if region1[k.strip()] != region2[k.strip()]:
                uniondiff += 1

        # +1 for each key not in both
        setsymmetricdiff = set1 ^ set2
        symmetricdiff = len(setsymmetricdiff)

        # Keep track of total
        total_distance += uniondiff + symmetricdiff

    return total_distance


def measure_distance_lev1(item1: str, item2: str, **kwargs):
    """
    Measure distance levenshtein (Polyleven library)
    """
    return levenshtein(item1, item2)


def measure_distance_lev2(record1: Any, record2: Any,
                          record1_kwargs: Dict, record2_kwargs: Dict,
                          env_kwargs: Optional[Dict] = None) -> int:
    """
    Levenshtein with support for concatenation of multiple fields.

    NOTE: Any NA values (pandas._libs.missing.NAType) will be treated
          as zero-length strings.

    Parameters
    ----------
    record1 : Any
        Dict-like record.

    record2 : Any
        Dict-like record.

    record1_kwargs : Any
        Dictionary with a 'columns' property containing
        a sequence of string properties to concatenate.
        e.g. ['cdr1_aa', 'cdr2_aa']

    record2_kwargs : Any
        Dictionary with a 'columns' property containing
        a sequence of string properties to concatenate.
        e.g. ['cdr1_aa', 'cdr2_aa']

    env_kwargs : Any, optional
        Environment-level arguments.
        Not used, default is None.

    Returns
    -------
    int
        The Levenshtein distance for the
        concatentated column values.
    """

    # Compute levenshtein - if NA type, convert to '' to avoid error.
    # (NA types may be present in cdr*_aa fields if residue annotation was not fully possible;
    #  these records should potentially be filtered out during analysis unless we wish to view them).
    return levenshtein(
        ''.join([record1[x] if not isinstance(record1[x], pd._libs.missing.NAType) else '' for x in record1_kwargs['columns']]),
        ''.join([record2[y] if not isinstance(record2[y], pd._libs.missing.NAType) else '' for y in record2_kwargs['columns']])
    )


# %%
def create_distance_matrix(records: pd.DataFrame, distance_function: Callable, record_kwargs: Dict, env_kwargs: Dict):
    """
    Compute square distance matrix for all pairs of records

    Only computes a pair once (but stores twice in square matrix).

    :param records: DataFrame of records, index will be used as headers

    :param distance_function: function that takes (a, b) and returns\
    a numeric distance

    :param record_kwargs: record-level kwargs for the distance function

    :param env_kwargs: environment-level kwargs for the distance function

    :returns: square DataFrame of distances
    """

    # Fix order of keys
    # keys = list(records)
    keys = list(records.index)

    # Create empty dataframe with headers
    df_distances = pd.DataFrame(index=keys, columns=keys)

    # Get all pairs (order doesn't matter)
    key_pairs = list(itertools.combinations(keys, 2))
    for pair in tqdm(key_pairs, desc='Create distance matrix'):
        key1, key2 = pair
        record1 = records.loc[key1]
        record2 = records.loc[key2]

        dist = distance_function(
            record1, record2, record_kwargs, record_kwargs, env_kwargs)
        df_distances.at[key1, key2] = dist
        df_distances.at[key2, key1] = dist

    # Distance between pairs of same key (key1, key1)
    # will not be provided by itertools.combinations
    # leaving these cells in the distance matrix as NaN.
    # So replace all NaNs with 0.
    assert sum(df_distances.isna().sum()) == len(keys), \
        'Expecting number of NaNs to be same as number of keys'
    df_distances.fillna(value=0, inplace=True)

    # Return
    return df_distances


# %% Verify triangle inequality
def verify_triangle_inequality(df_distances, n):
    """
    Random test a given sample size to ensure the
    triangle inequality holds. Chooses 3 random
    pairwise distances and ensure a + b >= c.

    :param df_distances: Distance Matrix
    :param n: Number of random tests to perform
    :returns: List of tuples containing indexes that fail
    """
    keys = list(df_distances.index)

    results = []

    for _ in tqdm(list(range(n)), desc="Verify triangle inequality..."):
        x = random.sample(keys, 3)
        a = df_distances.loc[x[0]][x[1]]
        b = df_distances.loc[x[0]][x[2]]
        c = df_distances.loc[x[1]][x[2]]
        if (a + b < c):
            results.append(tuple(x))

    return results


def compute_xy(distmatrix: pd.DataFrame,
               method: str,
               random_state: int) -> pd.DataFrame:
    """Compute xy sequence coordinates from dissimilarity matrix

    Args:
        distmatrix (pd.DataFrame): Square matrix of sequence distances.
            The .index property should be set, as this will be used in
            the returned DataFrame.

        method (str): PCA, TSNE or MDS

        random_state (int): Random state to be passed to PCA/TSNE/MDS fitting
            (reproducible results).

    Raises:
        Exception: If an invalid method is passed.

    Returns:
        pd.DataFrame: 'index', 'x', 'y' columns.
    """

    # Compute coordinates (PCA/TSNE/MDS)
    coords = None
    if method == 'PCA':
        model = PCA(n_components=2, random_state=random_state)
        coords = model.fit_transform(distmatrix)
    elif method == 'TSNE':
        model = manifold.TSNE(n_components=2, random_state=random_state)
        coords = model.fit_transform(distmatrix)
    elif method == 'MDS':
        model = manifold.MDS(n_components=2,
                             dissimilarity="precomputed",
                             random_state=random_state)
        coords = model.fit(distmatrix).embedding_  # [[x,y]] coordinates
    else:
        raise Exception(f"Invalid method: '{method}'")

    # Return xy coordinates
    xycoords = pd.DataFrame(coords,
                            index=list(distmatrix.index),
                            columns=['x', 'y'])
    xycoords.index.name = 'index'

    return xycoords


def compute_xy_distmatrix(xycoords: pd.DataFrame,
                          distmatrix: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise distance matrix from xy coordinates

    This returns a DataFrame in the same format as distmatrix, which
    can be used to compare the plotted distance to the original
    computed distance.

    Args:
        xycoords (pd.DataFrame): DataFrame containing index, 'x', 'y'
        distmatrix (pd.DataFrame): The original distance matrix. This
            is just used to get the index and column list.

    Returns:
        pd.DataFrame: Square dissimilarity matrix, with pairwise
            Euclidean distances computed from xy coordinates.
    """

    # Condensed distance matrix from xy points
    xy_distmatrix = pdist(xycoords[['x', 'y']])

    # Convert condensed matrix to square matrix + add headers
    xy_distmatrix = squareform(xy_distmatrix)
    assert (xy_distmatrix.shape == distmatrix.shape)

    # Load to DataFrame, set index and columns
    xydistmatrix_df = pd.DataFrame(xy_distmatrix,
                                   index=list(distmatrix.index),
                                   columns=list(distmatrix.columns))
    assert (xydistmatrix_df.shape == distmatrix.shape)

    # Return
    return xydistmatrix_df


def compute_delta(distmatrix: pd.DataFrame,
                  xy_distmatrix: pd.DataFrame) -> pd.DataFrame:
    """Sequence distance vs xy/plotted distance"""
    return xy_distmatrix - distmatrix


def melt_distmatrices(distmatrix: pd.DataFrame,
                      xy_distmatrix: pd.DataFrame,
                      delta_matrix: pd.DataFrame) -> pd.DataFrame:
    """Melt original, xy and delta matrices into one long table

    Args:
        distmatrix (pd.DataFrame): Square sequence distance matrix
        xy_distmatrix (pd.DataFrame): Square xy distance matrix
        delta_matrix (pd.DataFrame): Square delta matrix (sequence vs xy distance)

    Returns:
        pd.DataFrame: Long DataFrame, columns 'item1, item2,
            orig_distance, plot_distance, dist_delta'
    """

    # Convert to tall tables
    melt_distmatrix = distmatrix.reset_index().melt(
        id_vars='index', var_name='item2', value_name='orig_distance')
    melt_distmatrix.set_index(['index', 'item2'], inplace=True)

    melt_xy_distmatrix = xy_distmatrix.reset_index().melt(
        id_vars='index', var_name='item2', value_name='plot_distance')
    melt_xy_distmatrix.set_index(['index', 'item2'], inplace=True)

    melt_delta_matrix = delta_matrix.reset_index().melt(
        id_vars='index', var_name='item2', value_name='dist_delta')
    melt_delta_matrix.set_index(['index', 'item2'], inplace=True)

    # Join all three by item1 and item2 indexes
    df_melted = melt_distmatrix.join(
        melt_xy_distmatrix, how='left', sort=True)
    df_melted = df_melted.join(melt_delta_matrix, how='left', sort=True)
    df_melted.rename(columns={'index': 'item1'},
                     inplace=True)  # rename index -> item1

    # Return
    return df_melted
