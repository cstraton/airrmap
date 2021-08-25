# Add a new property to a parquet file

# %% Imports
import pandas as pd
import os
import io
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from typing import Any, Dict, Callable, List, Optional, Sequence

from airrmap.application.config import AppConfig, SeqFileType


def run_add_parquet_property(
        env_name: str,
        file_type: SeqFileType,
        transform_func: Callable,
        transform_kwargs: Dict = None,
        output_subfolder_name: str = 'modified',
        compression: Optional[str] = None,
        chunk_size: Optional[int] = None):
    """
    Add custom properties to sequence record or meta files.

    NOTE: Files in subfolder are overwritten, but not currently
    deleted between each run. Files will also need to be copied
    from the modified folder to the parent folder in order to
    take effect.

    Parameters
    ----------
    env_name : str
        Name of environment.

    file_type : SeqFileType
        Type of sequence file to process; RECORD or META.

    transform_func : Callable
        A transform function that takes a Pandas DataFrame row
        and any additional arguments provided with transform_kwargs.
        A dictionary should be returned, where keys will be used
        for the new column names. Consider using a prefix to
        avoid conflicting with existing column names.

    transform_kwargs : Dict, optional
        Dictionary of keyword arguments to pass to transform_func. By default None.

    output_subfolder_name : str, optional
        The name of the subfolder where the modified files will
        be written to, name only (not full path). This will be appended
        to the existing records subfolder, e.g. 'records_modified'.
        By default, 'modified'.

    compression : Optional[str], optional
        Compression type for written Parquet file; 'snappy' or 'gzip'.
        Pass None to use setting in envconfig.yaml. By default, None.

    chunk_size : Optional[int], optional
        Number of records to process at a time. Pass None to
        use setting in envconfig.yaml. By default, None.

    Raises
    ------
    Exception
        If sequence folder not found / doesn't exist.

    Exception
        If output subfolder name is not supplied.
    """

    appcfg = AppConfig()
    envcfg = appcfg.get_env_config(env_name)
    seqcfg = envcfg['sequence']

    # Default to env settings if not supplied
    compression = seqcfg['output_file_compression'] if compression is None else compression
    chunk_size = seqcfg['process_chunk_size'] if chunk_size is None else chunk_size

    folder = appcfg.get_seq_folder(
        env_name=env_name,
        file_type=file_type
    )
    if not os.path.exists(folder):
        raise Exception(f"Seq folder doesn't exist: {folder}")

    if output_subfolder_name.strip() == '':
        raise Exception('Output subfolder name is not supplied.')

    # Create the subfolder
    subfolder_path = f'{folder}_{output_subfolder_name}'
    if not os.path.isdir(subfolder_path):
        os.makedirs(subfolder_path, exist_ok=False)

    # Get list of files
    seq_files = appcfg.get_seq_files(env_name, file_type)

    # Process each one, write to output subfolder
    for fn_in in tqdm(seq_files):
        fn_name = os.path.basename(fn_in)
        fn_out = os.path.join(subfolder_path, fn_name)

        add_parquet_property(
            fn_in=fn_in,
            fn_out=fn_out,
            transform_func=transform_func,
            transform_kwargs=transform_kwargs,
            compression=compression,
            chunk_size=chunk_size
        )


def add_parquet_property(
        fn_in: Any,
        fn_out: Any,
        transform_func: Callable,
        transform_kwargs: Dict = None,
        compression: str = 'snappy',
        chunk_size: int = 1024):
    """
    Add new properties to a Parquet data file.

    Parameters
    ----------
    fn_in : str or file-like
        Input file path of the existing Parquet file.

    fn_out : str or file-like
        Ouput file path for the modified Parquet file.

    transform_func : Callable
        A transform function that takes a Pandas DataFrame row
        and any additional arguments provided with transform_kwargs.
        A dictionary should be returned, where keys will be used
        for the new column names. Consider using a prefix to
        avoid conflicting with existing column names.

    transform_kwargs : Dict, optional
        Additional keyword arguments to pass to transform_func,
        or None (default).

    compression : str, optional
        Compression type, 'snappy' or 'gzip. By default 'snappy'.

    chunk_size : int, optional
        Number of rows to process at a time, by default 1024.
    """

    # Check fn_out is not the same as fn_in
    if not isinstance(fn_in, io.BytesIO) and not isinstance(fn_out, io.BytesIO):
        if os.path.exists(fn_in) and os.path.exists(fn_out):
            if os.path.samefile(fn_in, fn_out):
                raise ValueError('File out cannot be the same as file in.')
        elif fn_in == fn_out:
            raise ValueError('File out cannot be the same as file in.')

    # Init
    transform_kwargs = {} if transform_kwargs is None else transform_kwargs

    # Init progress
    parquet_file = pq.ParquetFile(fn_in)
    record_count = parquet_file.metadata.num_rows
    #pbar = tqdm(total=record_count,
    #            desc=f'Reading {fn_in}', position=0, leave=True)

    # Open and loop through
    i = 0
    for pq_chunk in parquet_file.iter_batches(batch_size=chunk_size):
        df_chunk = pq_chunk.to_pandas()

    # with pd.read_parquet(
    #        path=fn_in,
    #        chunksize=chunk_size) as reader:

   #     for i, df_chunk in enumerate(reader):
        #pbar.update(len(df_chunk.index))

        # Compute additional properties
        # df_computed will be just the computed columns
        df_computed = df_chunk.apply(
            # df_computed = df_chunk.parallel_apply(
            transform_func,
            axis='columns',
            result_type='expand',
            **transform_kwargs
        )

        # Add the computed columns
        df_chunk = pd.concat(
            [df_chunk, df_computed],
            axis='columns'
        )

        # Convert to PyArrow table
        pqtable_chunk = pa.Table.from_pandas(df_chunk)

        # Create Parquet file
        if i == 0:
            pqwriter = pq.ParquetWriter(
                fn_out,
                pqtable_chunk.schema,
                compression=compression
            )

        # Append chunk to Parquet file
        pqwriter.write_table(pqtable_chunk)

        # Increase chunk counter
        i += 1

        # Close the parquet writer
    if pqwriter:
        pqwriter.close()
