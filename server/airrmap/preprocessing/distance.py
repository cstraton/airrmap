"""Distance measures between receptors"""

# tests > test_distance.py

# %% Imports
import pandas as pd
import numpy as np
import itertools
import random
from polyleven import levenshtein

from tqdm import tqdm
from sklearn import manifold
from sklearn.decomposition import PCA
from scipy.spatial import distance as scipydist
from scipy.spatial.distance import pdist, squareform
from typing import Callable, List, Tuple, Any


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


def measure_distance1(string1: str, string2: str):
    """
    Measure distance 1.

    Compare each character position
    starting from the first character.
    Distance is +1 for every different
    or different or additional character.
    """

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


def measure_distance3(item1: dict, item2: dict, regions: list):
    """
    Measure distance 3.

    Compute the distance between two
    annotated sequences. +1 for each residue different.
    CARE!: +1 distance will be added to any gaps - consider this if using different
    lengths / partials, or if the annotated sequence contains '.' items.

    :param item1: The sequence. Should be a dictionary, e.g.:
    record {} -> region ('cdrh2') -> residue number (imgt '26').

    :param item2: Second annotated sequence.
    :param regions: List of regions to compare. Each item must contain
    all regions passed (otherwise exception occurs).

    """

    # Ensure at least one region
    if len(regions) < 1:
        raise Exception('No regions provided')

    total_distance = 0
    for region_key in regions:
        region1 = item1[region_key]
        region2 = item2[region_key]
        set1 = set(region1.keys())
        set2 = set(region2.keys())

        # Count differences where a residue is in
        # both sets
        setintersect = set1.intersection(set2)
        uniondiff = 0
        for k in setintersect:
            if region1[k] != region2[k]:
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


def measure_distance_lev2(item1: Any, item2: Any, columns: List[str]) -> int:
    """
    Levenshtein with support for concatenation of multiple fields.

    Parameters
    ----------
    item1 : Any
        Dict-like record.

    item2 : Any
        Dict-like record.

    columns : List[str]
        List of string columns to concatenate
        in item1 and item2.

    Returns
    -------
    int
        The Levenshtein distance for the
        concatentated column values.
    """

    return levenshtein(
        ''.join([item1[x] for x in columns]),
        ''.join([item2[y] for y in columns])
    )


# %%
def create_distance_matrix(records: pd.DataFrame, distance_function: Callable,
                           measure_value: str = 'aa', **kwargs):
    """
    Compute square distance matrix for all pairs of records

    Only computes a pair once (but stores twice in square matrix).

    :param records: DataFrame of records, index will be used as headers

    :param distance_function: function that takes (a, b) and returns\
    a numeric distance

    :param measure_value: record property to measure distance for

    :param **kwargs: additional args required for the distance function

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
        # record1 = records[key1]
        # record2 = records[key2]
        record1 = records.loc[key1]
        record2 = records.loc[key2]

        dist = distance_function(
            record1[measure_value], record2[measure_value], **kwargs)
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
