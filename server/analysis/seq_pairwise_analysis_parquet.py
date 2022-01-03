# Contour map of sequence vs plot distance.

# Loads in the selected environment.
# Selects 2n random records with replacement.
# Splits in two to create random pairs
# Computes sequence distance and
# Euclidean distance for each pair.

# USE:
# 1. Set CONFIG params.
# 2. Run all, will write png image to current folder.

# %% Imports
import plotly.express as px
import pandas as pd
import json
import os
import math
from plotly.graph_objs._figure import Figure  # Reference to Figure
from typing import Dict, List, Tuple, Union
from tqdm import tqdm

import airrmap.preprocessing.distance as distance
import airrmap.preprocessing.multilateration as multilateration
from airrmap.application.config import AppConfig, SeqFileType

# %% CONFIG
# %% Main
#env_name = 'Eliyahu_CDRH3_Lev1'
env_name = 'GUPTA_CDRH3_LEV1_gapped'
num_pairs = 50000
out_name = f'{env_name}_{num_pairs}.png'


# %% Read files from Parquet
#    (records and coordinates all included)


def get_df_dists(env_name: str,
                 num_pairs: int) -> pd.DataFrame:
    """
    Generate dataframe of distances between pairs of sequences.

    Parameters
    ----------
    env_name : str
        Environment name.
    num_pairs : int
        Number of pairs of random sequence pairs.

    Returns
    -------
    DataFrame
        Pandas DataFrame, containing pairs of points,
        distances and error.
    """

    # Get the environment configuration
    appcfg = AppConfig()
    envcfg = appcfg.get_env_config(env_name)
    seqcfg = envcfg['sequence']
    anccfg = envcfg['anchor']
    distance_measure = envcfg['distance_measure']
    regions = envcfg['distance_measure_options']['regions']
    distance_function = getattr(distance, distance_measure)
    random_state = anccfg['random_state']
    seq_column = seqcfg['seq_field']

    # Get Parquet record files
    seq_files = appcfg.get_seq_files(
        env_name=env_name,
        file_type=SeqFileType.RECORD
    )

    # Columns to read
    columns = [
        'sys_point_id',
        'sys_coords_x',
        'sys_coords_y',
        seq_column
    ]

    # Read to dataframe
    df = pd.read_parquet(
        path=seq_files,
        columns=columns
    )

    # %% Get samples
    # Allow replace=True in case
    # not enough samples.
    # Currently weights=None,
    # as we may not always be interested
    # in measuring the same value (e.g. redundancy).
    # Also weighting shifts depending on how
    # we have the data filtered.
    df_sample = df.sample(
        n=num_pairs * 2,
        replace=True,
        weights=None,
        random_state=random_state
    )

    # %% Split into two DataFrames
    #    (each side of a pair)
    df1 = df_sample.iloc[:num_pairs, :]
    df2 = df_sample.iloc[num_pairs:, :]

    # %% Generate DataFrame of Plotted vs. Sequence distances
    dist_records: List[Tuple] = []
    for i in tqdm(range(num_pairs), desc=f'Creating {num_pairs} samples...'):

        # Get id, IMGT numbered_sequence ('data')
        # and xy coords
        point1 = df1.iloc[i]
        point1_id = point1['sys_point_id']
        point1_seq = point1[seq_column]
        point1_x = point1['sys_coords_x']
        point1_y = point1['sys_coords_y']

        point2 = df2.iloc[i]
        point2_id = point2['sys_point_id']
        point2_seq = point2[seq_column]
        point2_x = point2['sys_coords_x']
        point2_y = point2['sys_coords_y']

        # Compute sequence distance
        seq_dist = distance_function(
            point1_seq,
            point2_seq,
            regions=regions
        )

        # Compute plotted distance
        xy_dist = distance.compute_euclidean(
            point1_x,
            point1_y,
            point2_x,
            point2_y
        )

        # Add results
        dist_records.append(
            (point1_id, point1_seq, point2_id, point2_seq, seq_dist, xy_dist)
        )

    # %% Create df of distances
    columns_dist = ['point1_id', 'point1_seq', 'point2_id',
                    'point2_seq', 'orig_distance', 'plot_distance']
    df_dists = pd.DataFrame.from_records(
        data=dist_records,
        columns=columns_dist
    )

    return df_dists


def plot_density_contour(df_dists: pd.DataFrame):
    fig1: Figure = px.density_contour(
        df_dists,
        x='orig_distance',
        y='plot_distance',
        labels={
            "orig_distance": "Sequence distance",
            "plot_distance": "Plot distance"

        }
    )

    fig1.update_traces(
        contours_coloring="fill",
        contours_showlabels=True
    )
    fig1.update_traces(
        showscale=False
    )

    # return
    return fig1


def compute_add_mse(df_dists: pd.DataFrame):
    """Compute means squared error, and add to df"""
    df_dists['err'] = df_dists.apply(lambda row: (
        row['plot_distance'] - row['orig_distance']), axis=1)
    df_dists['err_sqr'] = df_dists['err'].apply(lambda x: x**2)
    mse = df_dists['err_sqr'].sum() / len(df_dists.index)
    rmse = math.sqrt(mse)
    return mse, rmse


# %% Get dataframe of distances
df_dists = get_df_dists(
    env_name=env_name,
    num_pairs=num_pairs
)

# %% Add on error
mse, rmse = compute_add_mse(df_dists)


# %% Get figures
fig1 = plot_density_contour(
    df_dists
)

# %% Update layout
fig1.update_xaxes(range=[0, 26])
fig1.update_yaxes(range=[0, 26])
fig1.update_layout(
    width=600,
    height=600,
    template='simple_white',
    font=dict(
        size=16
    )
)

# %% Show
# fig1.show()
# fig2.show()

# %% Save
fig1.write_image(out_name)

# %%
# df_dists['plot_distance'].hist()
# %%
# df_dists['orig_distance'].hist()
