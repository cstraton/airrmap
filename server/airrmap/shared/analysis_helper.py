# Helper functions for analyses

import os
import sys
import sqlite3
import pandas as pd
from typing import List, Dict, Union

from airrmap.application.repo import DataRepo
from airrmap.application.config import AppConfig, SeqFileType
from airrmap.application import tile
from airrmap.application import faststats_kde


def get_repo_cfg():
    """Return DataRepo and AppConfig in one call."""
    cfg = AppConfig()
    return DataRepo(cfg), cfg


def get_query(env_name: str,
              facet_row: str = '',
              facet_col: str = '',
              filters: List[Dict[str, str]] = []):
    """
    Get a query instance.

    Parameters
    ----------
    env_name : str, optional
        Environment name, by default 'Anchor_MLAT_Analysis'

    facet_row : str, optional
        Field name for facet row, by default ''

    facet_col : str, optional
        Field name for facet column, by default ''

    filters : List[Dict[str, str]], optional
        List of filters to apply, by default [].
        Example:
        [
            {'filter_name': 'f.Chain', 'filter_value': 'Heavy'}
        ]


    Returns
    -------
    Dict
        The query instance.
    """

    # (Adjusted from the docstring for run_coords_query)
    query = dict(
        env_name=env_name,
        facet_row=facet_row,
        facet_col=facet_col,
        filters=filters
    )

    return query


def load_anchors(env_name: str, appcfg: AppConfig = None):
    """Load anchors for the given environment (records and coords)"""
    appcfg = AppConfig() if appcfg is None else appcfg
    anchors_db = appcfg.get_anchordb(env_name)

    sql_anchors = """
        SELECT rec.*, coords.x AS x, coords.y AS y
        FROM anchor_records AS rec
        LEFT JOIN anchor_coords AS coords
        ON rec.anchor_id = coords.anchor_id 
    """

    with sqlite3.connect(f'file:{anchors_db}?mode=ro', uri=True) as conn:
        df_anchors = pd.read_sql(sql_anchors, con=conn)

    return df_anchors


def load_sequences(env_name: str,
                   columns=None,
                   appcfg: AppConfig = None,
                   filters=None) -> pd.DataFrame:
    """
    Load sequences directly from files.

    Parameters
    ----------
    env_name : str
        The name of the environment.

    columns : List, optional
        The columns to load, None for all (default).

    filters : List[Tuple] or List[List[Tuple]] or None, optional
        Filters for the PyArrow Parquet read_table method. e.g.: 
        `('x', 'in', ['a', 'b', 'c'])`.
        Default is None. 
        See [https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html]

    appcfg : AppConfig, optional
        Application configuration, by default None (use default).

    Returns
    -------
    pd.DataFrame
        The sequence records.
    """

    # Get application config
    appcfg = AppConfig() if appcfg is None else appcfg

    # Get list of record files in the environment
    seq_file_list = appcfg.get_seq_files(
        env_name=env_name,
        file_type=SeqFileType.RECORD
    )

    # If only one file, take out of array
    # (otherwise Pandas/PyArrow error.)
    seq_file_list = seq_file_list[0] if len(
        seq_file_list) == 1 else seq_file_list  # type: ignore

    # Read to DataFrame
    df: pd.DataFrame = pd.read_parquet(
        seq_file_list,
        columns=columns,
        filters=filters
    )

    # Return
    return df
