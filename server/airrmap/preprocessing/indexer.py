"""
Metadata Index functionality.

Loop through the data environments
and sequence meta and record files. Open each one and
collect the meta table information and
list of columns in the records table, then
write to index.db.
Overwrites previous results.
"""

# %% Imports
import pyarrow.parquet as pq
import sqlescapy
import pathlib
import logging
import sqlite3
import pandas as pd
import sys
from tqdm import tqdm
from typing import List, Dict

from airrmap.application.config import AppConfig
from airrmap.application.config import SeqFileType


def _recur_dictify(df: pd.DataFrame) -> Dict:
    """
    Converts Pandas DataFrame to -nested- dictionary

    First 3 columns are grouped by, last column for values.
    """
    # Adapted from DSM, November 2013:
    # https://stackoverflow.com/questions/19798112/convert-pandas-dataframe-to-a-nested-dict

    if len(df.columns) == 1:
        if df.values.size == 1:
            return df.values[0][0]
        return df.values.squeeze()
    grouped = df.groupby(df.columns[0])
    d = {k: _recur_dictify(g.iloc[:, 1:]) for k, g in grouped}
    return d

# %%


def build_index(cfg: AppConfig):
    """Read metadata and write to index db"""

    envs: List[str] = cfg.get_envs()
    env_meta: List[pd.DataFrame] = []

    for env_name in tqdm(envs, 'Reading sequence meta and record files...'):

        # Read meta files
        seq_meta_files = cfg.get_seq_files(env_name, SeqFileType.META)
        if len(seq_meta_files) == 0:
            raise Exception(
                f"No meta files found for environment '{env_name}'")
            break

        for fn_seq_meta in seq_meta_files:
            df_meta = read_seq_meta(env_name, fn_seq_meta)
            env_meta.append(df_meta)

        # Read record files (just the available columns, not all the records)
        seq_record_files = cfg.get_seq_files(env_name, SeqFileType.RECORD)
        if len(seq_record_files) == 0:
            raise Exception(
                f"No record files found for environment '{env_name}'")
            break

        for fn_seq_record in seq_record_files:
            df_recordcols = read_seq_recordcols(env_name, fn_seq_record)
            env_meta.append(df_recordcols)

    # Save results
    assert cfg.index_db != ''
    print(f'Writing to index database ({cfg.index_db})...')
    with sqlite3.connect(cfg.index_db) as conn:

        TABLE_NAME = 'env_index'

        # Drop existing table
        sql_drop = f"""
            DROP TABLE IF EXISTS {TABLE_NAME}
        """
        conn.execute(sql_drop)

        # Write DataFrames
        for df_meta in env_meta:
            df_meta.to_sql(
                TABLE_NAME,
                conn,
                if_exists='append',
                index=False
            )

    # Raise error if nothing found
    if len(env_meta) == 0:
        raise Exception('No sequence files found, check config.')

    # Completed
    print('Index build completed.')


def read_seq_meta(env_name: str,
                  fn_seq_meta: str) -> pd.DataFrame:
    """Read metadata Parquet file"""

    # Sql and escape env and data unit name (folder/files)
    file_name = pathlib.Path(fn_seq_meta).name

    # Replace ext .meta.parquet to .parquet, to match the
    # filename used by the records file.
    # e.g. file.meta.parquet -> file.parquet
    suffix = '.meta.parquet'
    if file_name.endswith(suffix):
        file_name = file_name[:len(file_name) - len(suffix)] + '.parquet'

    # Read to dataframe
    try:
        df: pd.DataFrame = pd.read_parquet(fn_seq_meta)
    except Exception as ex:
        logging.exception(f"Error reading meta file: '{fn_seq_meta}'")
        sys.exit(1)

    # Add env name and filename
    df.insert(0, 'env_name', env_name)
    df.insert(1, 'file_name', file_name)
    df.insert(2, 'property_level', 'file')
    df.insert(4, 'property_type', 'TEXT')

    # Return
    return df


def read_seq_recordcols(env_name: str,
                        fn_seq_record: str) -> pd.DataFrame:
    """Read list of columns in Parquet seq records file.

    Only field names are retrieved, not list of values
    (too many to list).
    """

    # Sql and escape env and data unit name (folder/files)
    file_name = pathlib.Path(fn_seq_record).name

    # Read Parquet schema (doesn't load whole file)
    try:
        schema = pq.read_schema(fn_seq_record)
    except Exception as ex:
        logging.exception(f"Error reading seq record file: '{fn_seq_record}'")
        sys.exit(1)

    # Get the columns
    field_list = []

    # Build up list of property records
    # Go through each field in schema
    for i in range(len(schema)):
        field = schema[i]
        field_type = str(field.type)
        field_name = field.name
        record = dict(
            env_name=env_name,
            file_name=file_name,
            property_level='record',
            property_name=field_name,
            property_type=field_type,
            property_value=''
        )
        field_list.append(record)

    # Create DataFrame
    df = pd.DataFrame.from_records(data=field_list, index=None)

    # Return
    return df


def get_index(fn_indexdb: str) -> Dict:
    """Get index data as dictionary"""

    # { env_name: {property_name: property_value}}

    sql_select = """
        SELECT      env_name,
                    file_name,
                    property_name,
                    property_value

        FROM        env_index
        ORDER BY    env_name, file_name, property_name
    """

    # Read index table
    with sqlite3.connect(fn_indexdb) as conn:
        df: pd.DataFrame = pd.read_sql(
            sql_select,
            conn,
            index_col=None,
        )

    # Convert to nested dictionary
    result = _recur_dictify(df)

    return result


def get_env_list(fn_indexdb: str) -> List:
    """Get the list of environments and details"""

    sql_select = """
        SELECT      MIN(env_name) as env_name,
                    COUNT(DISTINCT file_name) AS total_file_count,
                    SUM(CASE WHEN property_name='sys_file_size' THEN CAST(property_value as integer) ELSE 0 END) AS source_file_size,
                    SUM(CASE WHEN property_name='sys_records' THEN CAST(property_value as integer) ELSE 0 END) AS total_records,
                    MAX(CASE WHEN property_name='sys_processed_utc' THEN property_value ELSE  '' END) AS last_processed_utc

        FROM        env_index
        WHERE       property_level = 'file'
        GROUP BY    env_name
        ORDER BY    env_name
    """

    ENV_NAME = 0
    FILE_COUNT = 1
    SOURCE_FILE_SIZE = 2
    RECORD_COUNT = 3
    LAST_PROCESSED = 4

    with sqlite3.connect(fn_indexdb) as conn:
        curr = conn.cursor()
        result = curr.execute(sql_select).fetchall()
        result_d = [
            dict(name=x[ENV_NAME],
                 total_file_count=x[FILE_COUNT],
                 source_file_size=x[SOURCE_FILE_SIZE],
                 total_records=x[RECORD_COUNT],
                 last_processed_utc=x[LAST_PROCESSED])
            for x in result
        ]
        curr.close()

    return result_d


def get_filter_item(fn_indexdb: str,
                    env_name: str,
                    filter_name: str) -> Dict:
    """Get a filter item and valid selection options.

    Parameters
    ----------
    fn_indexdb : str
        Path of the index database.

    env_name : str
        Environment name.

    filter_name : str
        Filter name, e.g. record.v

    Returns
    -------
    Dict
        Item in following format:
        {"name": "filter.name",
         "value_type": "TEXT",
         "options": [
             "key": "option1key"
             "text": "option1text"
             "value": "option1value"
            ]
        }
    """

    sql_select = """
        SELECT      property_value as filter_value,
                    property_type as filter_value_type
        FROM        env_index
        WHERE       env_name = ?
          AND       property_level = ?
          AND       property_name = ?
        GROUP BY    property_value
        ORDER BY    property_value
    """

    PROPERTY_VALUE = 0
    PROPERTY_TYPE = 1

    property_level, property_name = filter_name.split('.')
    params = [env_name, property_level, property_name]

    with sqlite3.connect(fn_indexdb) as conn:
        curr = conn.cursor()
        result = list(curr.execute(sql_select, params).fetchall())

        # First record, type should be same for all records
        value_type = result[0][PROPERTY_TYPE]

        option_list = [dict(key=x[PROPERTY_VALUE],
                            text=x[PROPERTY_VALUE],
                            value=x[PROPERTY_VALUE])
                       for x in result if len(x[PROPERTY_VALUE]) > 0]
        curr.close()

    result_d = dict(name=filter_name,
                    value_type=value_type,
                    options=option_list)

    return result_d


def get_filters(fn_indexdb: str,
                env_name: str) -> List[Dict]:
    """The list of file/record properties that can be filtered on."""

    sql_select = f"""
        SELECT  env_index.property_level || '.' || env_index.property_name as property_name
        FROM    env_index
        WHERE   env_name = ?
        GROUP BY property_name
        ORDER BY property_name
    """

    # Read
    with sqlite3.connect(fn_indexdb) as conn:
        df: pd.DataFrame = pd.read_sql(
            sql_select,
            conn,
            index_col=None,
            params=[env_name]
        )
    property_names = list(df['property_name'])

    # Format to [{key, text, value}] for consumer
    # (May need to change key / values in future)
    property_names = [
        dict(key=x, text=x, value=x)
        for x in property_names
    ]

    return property_names
