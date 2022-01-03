# Sequence distance vs
# plotted distance for different
# numbers of anchors (box plot figure).

# Notes:
# 1. Varies number of anchors.
# 2. Zero-centered. Distance offset (euclidean vs sequence).
# 3. Select seq_sample_size number of sequences
# 4. Creates all-against-all pairwise distnaces.

# 1. To Run, set the config.
# 2. Run all.

# %% Imports
import yaml
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import math
import random
import numpy as np
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Tuple, Union

import airrmap.shared.analysis_helper as ah
import airrmap.preprocessing.imgt as imgt
import airrmap.preprocessing.distance as distance
from airrmap.application.config import AppConfig, SeqFileType
from airrmap.preprocessing.oas_adapter_base import OASAdapterBase, AnchorItem


# %% --- SET CONFIG HERE ---
seq_env_name = 'GUPTA_CDRH3_LEV1'
# Eliyahu_CDRH3_Lev1_GappedSeq
# GUPTA_CDRH3_LEV1
anchor_env_name = 'GUPTA_CDRH3_LEV1'
anchor_seq_field = 'cdr3'  # 'aa_annotated'
anchor_random_state = 22  # 22 Random state for anchor sample (int or None).

# Number of sequences to select
# Will measure all-against-all for each
# number of anchors.
# Suggest 200 max = 40,000 pairs for dist matrix
# (position of each point must be computed).
seq_sample_size = 200
# 3 Random state for sequence sample (should be None or different int to anchor_random_state).
random_state = 3

distance_measure_name = 'measure_distance_lev1'  # 'measure_distance3'
distance_measure_func = getattr(distance, distance_measure_name)
distance_measure_kwargs: Dict = {}  # dict(regions=['cdrh1', 'cdrh2'])
reduction_method = 'MDS'  # 'UMAP' 'MDS'
def anchor_seq_transform(x): return x  # json.loads
def seq_transform(x): return x  # json.loads


use_mae = False
#bootstap_num = 1000
num_anchors_list = [5, 10, 25, 50, 100, 250, 500]  # , 50, 100]


def get_seq_sample(env_name: str,
                   seq_sample_size: int,
                   seq_field: str,
                   random_state: int) -> pd.DataFrame:
    """
    Get a random sample of sequences.

    Reads all sys_point_id values into memory first 
    (less memory usage), and takes a sample (with replacement).
    Then loads in the seq_field values for the selected IDs.

    Parameters
    ----------
    env_name : str
        Name of the environment

    seq_sample_size : int
        Size of the sample to get.

    seq_field : str
        Name of the sequence field

    random_state : int
        Random state.

    Returns
    -------
    DataFrame:
        Sample of sequence records.
    """

    # %% First read in the list of point ids, then
    # select a sample, and load in the sequences.
    # (Don't load sequences directly, as large memory usage).
    # replace=False to avoid possible duplicate records (causes error).
    df_seqids = ah.load_sequences(env_name, columns=['sys_point_id'])
    seqids_list = list(
        df_seqids.sample(
            n=seq_sample_size,
            replace=False,
            weights=None,
            random_state=random_state
        )['sys_point_id']
    )

    # %% Load
    df_seqs = ah.load_sequences(
        env_name=env_name,
        columns=[seq_field],
        filters=[['sys_point_id', 'in', seqids_list]]
    )

    # Return
    return df_seqs


def get_coords_using_reduction(
        df_seqs: pd.DataFrame,
        seq_field: str,
        distance_measure_name: str,
        distance_measure_kwargs: Dict,
        reduction_method: str,
        random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute coords using all-against-all distance matrix and
    dimensionality reduction.

    Parameters
    ----------
    df_seqs : pd.DataFrame
        Sequences to use.

    seq_field : str
        Sequence field in the DataFrame.

    distance_measure_name : str
        Distance measure to use.

    distance_measure_kwargs : Dict
        Additional keyword arguments for the distance measure.

    reduction_method : str
        Dimensionality reduction method (e.g. 'MDS').

    random_state : int
        Random state to use.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Two DataFrames, one a list of coordinates and the
        other a list of distances (sequence and Euclidean).
    """

    # Build distance matrix
    df_dist_matrix = imgt.build_distance_matrix(
        df=df_seqs,
        distance_function=distance_measure_func,
        measure_value=seq_field,
        # Set to 10 for performance, tested already using this method.
        triangle_inequality_samples=10,
        **distance_measure_kwargs
    )

    # Compute the coords and distances
    df_coords, df_dist = imgt.compute_coords(
        df_dist_matrix=df_dist_matrix,
        method=reduction_method,
        random_state=random_state)

    # Return
    return df_coords, df_dist


# %% Compute coordinates for sequences
def get_coords_using_mlat(
        df_seqs: pd.DataFrame,
        seq_field: str,
        anchor_items: Dict[int, AnchorItem],
        num_closest_anchors: int,
        distance_measure_name: str,
        distance_measure_kwargs: Dict,
        use_mae: bool):
    """
    Compute the coordinates for given sequeunces using multilateration.

    Parameters
    ----------
    df_seqs : pd.DataFrame
        Selected sample of sequences.

    seq_field : str
        Field containing the sequence.

    anchor_items : Dict[int, AnchorItem]
        Loaded AnchorItem instances.

    num_closest_anchors : int
        Number of closest anchors to use.

    distance_measure_name : str
        Distance measure to use.

    distance_measure_kwargs : Dict
        Any arguments for the distance measure.

    use_mae : bool
        True to use mean absolute error instead of
        mean squared error during computation of
        sequence coordinates.

    Returns
    -------
    DataFrame
        The sample of sequences.
    List[Dict]
        List of dictionaries containing the computed
        coordinates and additional properties. Items
        are in the same order as the DataFrame sample.
    """

    # Compute coordinates
    results = []
    for i in tqdm(range(len(df_seqs.index))):
        seq_coords_d = OASAdapterBase.process_single_record(
            df_seqs.iloc[i],
            seq_field=seq_field,
            anchors=anchor_items,
            num_closest_anchors=num_closest_anchors,
            distance_measure_name=distance_measure_name,
            distance_measure_kwargs=distance_measure_kwargs,
            save_anchor_dists=True,
            use_mae=use_mae
        )
        results.append(seq_coords_d)

    # Verify
    assert len(results) ==  \
        len(df_seqs.index), \
        'Should be same number of sequences and coordinates.'

    # Return
    return df_seqs, results


def get_dist_delta(seq_sample_size: int,
                   df_seq_sample: pd.DataFrame,
                   df_seq_coords: pd.DataFrame,
                   distance_measure_func: Callable,
                   distance_measure_kwargs: Dict) -> pd.DataFrame:

    # Verify
    assert len(df_seq_sample.index) == len(df_seq_coords.index), \
        'Number of sequences and sequence coordinates should be the same.'

    # Generate all-against-all pairs
    index_pairs = list(
        itertools.product(
            range(seq_sample_size),
            range(seq_sample_size)
        )
    )

    # %% Generate all combinations
    pair_distances = []
    for i1, i2 in tqdm(index_pairs):

        # Select pair
        seq1 = df_seq_coords.iloc[i1]
        seq2 = df_seq_coords.iloc[i2]

        # Get coordinates
        x1 = float(seq1['sys_coords_x'])
        y1 = float(seq1['sys_coords_y'])
        x2 = float(seq2['sys_coords_x'])
        y2 = float(seq2['sys_coords_y'])

        # Compute distance
        euclidean_dist = distance.compute_euclidean(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2
        )

        # Compute sequence distance
        seq1_record = seq_transform(str(df_seq_sample.iloc[i1][seq_field]))
        seq2_record = seq_transform(str(df_seq_sample.iloc[i2][seq_field]))
        seq_dist = distance_measure_func(
            seq1_record, seq2_record, **distance_measure_kwargs)

        # Compute difference
        # Work out whether euclidean distance is
        # above or below sequence distance
        dist_delta = euclidean_dist - seq_dist

        # Add to the list
        pair_distances.append(
            dict(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                seq_dist=seq_dist,
                euclidean_dist=euclidean_dist,
                dist_delta=dist_delta
            )
        )

    # Return
    df_pair_distances = pd.DataFrame.from_records(pair_distances)
    return df_pair_distances


def get_anchor_items(env_name: str,
                     sample_size: int,
                     seq_field: str,
                     distance_measure_name: str,
                     distance_measure_kwargs: Dict,
                     reduction_method: str,
                     random_state: Optional[int]
                     ) -> Tuple[Dict[int, AnchorItem], pd.DataFrame, pd.DataFrame]:
    """
    Get random AnchorItems with computed coordinates.

    Parameters
    ----------
    env_name : str
        Environment name containing the sequences.

    sample_size : int
        Size of the sample (no replacement).

    seq_field : str
        Field containing the sequence.

    distance_measure_name : str
        Distance measure to use.

    distance_measure_kwargs : Dict
        Additional keyword arguments for the distance measure.

    reduction_method : str
        Dimensionality reduction method (e.g. 'MDS').

    random_state : int
        Random state to use

    Returns
    -------
    Dict[int, AnchorItem]
        Dictionary of Anchor Items.
    DataFrame
        Computed coordinates.
    DataFrame
        Distances (sequence and Euclidean).
    """

    # %% Get a sample of anchors
    df_anchors_sample = get_seq_sample(
        env_name=env_name,
        seq_sample_size=sample_size,
        seq_field=seq_field,
        random_state=random_state
    )

    # %% Compute anchor coordinates
    df_anchor_coords, df_anchor_dists = get_coords_using_reduction(
        df_seqs=df_anchors_sample,
        seq_field=anchor_seq_field,
        distance_measure_name=distance_measure_name,
        distance_measure_kwargs=distance_measure_kwargs,
        reduction_method=reduction_method,
        random_state=random_state
    )

    # %% Place in AnchorItem instances
    anchors_seq = list(df_anchors_sample[anchor_seq_field])
    anchors_id = list(df_anchor_coords.index)
    anchors_x = list(df_anchor_coords['x'])
    anchors_y = list(df_anchor_coords['y'])
    anchor_items = {anchor_id: AnchorItem(anchor_id=anchor_id, seq=seq, x=x, y=y)
                    for anchor_id, seq, x, y in zip(anchors_id, anchors_seq, anchors_x, anchors_y)}

    # Return
    return anchor_items, df_anchor_coords, df_anchor_dists


# %% Get environment settings
appcfg = AppConfig()
seqenvcfg = appcfg.get_env_config(seq_env_name)
seqcfg = seqenvcfg['sequence']
seq_field = seqcfg['seq_field']


# %% Load selected sequences
df_seqs_sample = get_seq_sample(
    env_name=seq_env_name,
    seq_sample_size=seq_sample_size,
    seq_field=seq_field,
    random_state=random_state
)


# %% Set up loop of number of anchors
dist_delta_num_anchors = {}

# Loop and plot
for num_anchors in num_anchors_list:

    print(f'Number of anchors: {num_anchors}')

    # %% Get sample of anchors with computed coordinates
    anchor_items, df_anchor_coords, df_anchor_dists = get_anchor_items(
        env_name=anchor_env_name,
        sample_size=num_anchors,
        seq_field=anchor_seq_field,
        distance_measure_name=distance_measure_name,
        distance_measure_kwargs=distance_measure_kwargs,
        reduction_method=reduction_method,
        random_state=anchor_random_state
    )

    # Get sample and computed coordinates
    df_seq_sample, seq_coords = get_coords_using_mlat(
        df_seqs=df_seqs_sample,
        seq_field=seq_field,
        anchor_items=anchor_items,
        num_closest_anchors=0,  # Use all anchors
        distance_measure_name=distance_measure_name,
        distance_measure_kwargs=distance_measure_kwargs,
        use_mae=use_mae
    )

    # %% Get DataFrame of coords
    df_seq_coords = pd.DataFrame.from_records(seq_coords)

    # %% Get differences between euclidean and sequence distance
    df_pair_distances = get_dist_delta(
        seq_sample_size=seq_sample_size,
        df_seq_sample=df_seq_sample,
        df_seq_coords=df_seq_coords,
        distance_measure_func=distance_measure_func,
        distance_measure_kwargs=distance_measure_kwargs
    )

    # Add result
    dist_delta_num_anchors[num_anchors] = list(
        df_pair_distances['dist_delta'])


# %% Get plot details
plot_labels = []
plot_data = []
for k, v in dist_delta_num_anchors.items():
    plot_labels.append(k)
    plot_data.append(v)
plot_data = np.array(plot_data).swapaxes(0, 1)  # type: ignore


# %% Try pure MDS (without anchors/multilateration, for comparison)
#df_mds_sample = df_seqs_sample.sample(n=int(len(df_seqs_sample) ** 0.5))
df_dist_matrix = imgt.build_distance_matrix(
    df=df_seqs_sample,
    distance_function=distance_measure_func,
    measure_value=seq_field,
    triangle_inequality_samples=10  # already tested
)

# %% Compute pure MDS results without anchors
df_mds_coords, df_mds_dist = imgt.compute_coords(
    df_dist_matrix=df_dist_matrix,
    method=reduction_method,
    random_state=random_state)

# %% MDS add to box plot
# Number of values will be seq_sample_size ^2.
# Get a sample of so we can plot alongside the original
mds_dist_values = np.expand_dims(
    df_mds_dist['dist_delta'].values, axis=1)
plot_data_final = np.hstack((plot_data, mds_dist_values))
plot_labels_final = plot_labels.copy()
plot_labels_final.append(reduction_method)


# Set up figure
plt.figure(figsize=(5, 5), dpi=200)
flierprops = dict(
    marker='o',
    markeredgecolor='#888888',
    markeredgewidth=0.1

)

# Box plot
sns.set_style('whitegrid')
plt.boxplot(
    plot_data_final,
    # bootstrap=bootstap_num,
    conf_intervals=None,
    showmeans=False,
    flierprops=flierprops,
    # notch=True
)

# Add labels
plt.xticks(range(1, len(plot_labels_final) + 1), plot_labels_final)
plt.xlabel('Number of Anchors')
plt.ylabel('Plot Distance Error')
# plt.grid()
plt.ylim([-25, 25])
