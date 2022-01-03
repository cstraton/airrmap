# JSON/CSV Adapter for OAS datasets.
# Calls multiple functions in the base
# class OASAdapterBase.
# Primarily used to compute the 2D embeddings
# for the study sequences.

# %% Imports
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import json
from collections import OrderedDict
from tqdm import tqdm
from typing import Any, Dict, List, Callable
from pandarallel import pandarallel

import airrmap.preprocessing.pointid as pointid
from airrmap.preprocessing.oas_adapter_base import OASAdapterBase, AnchorItem


# %% OASAdpaterJSON Class


class OASAdapterJSON(OASAdapterBase):

    @staticmethod
    def process_meta(file_id: int,
                     fn,
                     fn_out,
                     openfunc: Callable,
                     record_count: int, fn_anchors_sha1: str,
                     anchor_dist_compression,
                     anchor_dist_compression_level):

        FILE_ID_FIELD = 'sys_file_id'

        # Open and read meta
        with openfunc(fn) as f:
            header_json = next(f)
            header_dict = json.loads(header_json)

        # Get additional file properties
        # super(a, a) required for static method
        extra_properties = super(OASAdapterJSON, OASAdapterJSON).get_extra_file_meta(
            file_id=file_id,
            fn=fn,
            record_count=record_count)

        # Get anchor distance params
        extra_properties['sys_dist_compression'] = anchor_dist_compression
        extra_properties['sys_dist_compression_level'] = anchor_dist_compression_level
        extra_properties['sys_dist_anchor_db_sha1'] = fn_anchors_sha1

        # Convert to long DataFrame
        # # super(a, a) required for static method
        df_header = super(OASAdapterJSON, OASAdapterJSON).header_to_df(
            header=header_dict,
            extra=extra_properties
        )

        # Convert all values to string (otherwise Parquet will complain)
        df_header['property_value'] = df_header['property_value'].astype('str')

        # Add sys_file_id (will need to link to the records Parquet file in queries)
        df_header.insert(0, FILE_ID_FIELD, file_id)

        # Write out the parquet file
        df_header.to_parquet(fn_out, index=False)

    @staticmethod
    def process_records(file_id: int,
                        fn,
                        fn_out,
                        openfunc: Callable,
                        record_count: int,
                        seq_id_field: str,
                        seq_field: str,
                        anchors: Dict[int, AnchorItem],
                        num_closest_anchors: int,
                        distance_measure_name: str,
                        distance_measure_kwargs: Dict,
                        compression='snappy',
                        chunk_size=5000,
                        save_anchor_dists=False,
                        anchor_dist_compression='zlib',
                        anchor_dist_compression_level=9,
                        stop_after_n_chunks: int = 0,
                        nb_workers: int = None):
        """
        Process the sequence records in a data unit file.

        This method primarily computes the sequence distance between
        each sequence in the data unit and the list of anchors. From
        this, the sequence coordinates are computed.
        Processing is performed in chunks, with parallel support.


        Parameters
        ----------
        file_id : int
            The generated numeric file id for the data unit file.
            Should be unique within the environment.

        fn : str or file-like
            The data unit filename (full path).

        fn_out : str or file-like
            The file to save the processed data to.

        openfunc : Callable
            The function to open the data unit with.
            Should gzip.open() if a gzipped file, otherwise
            the built-in open() function.

        record_count : int
            Number of sequences/records in the data (excluding headers)

        seq_id_field : str or None
            The integer field in the data unit to use as the sequence ID,
            unique within the data unit file (a separate ID will be created
            to make unique within the environment).
            Specify None to auto-generate.

        seq_field : str
            The field containing the sequence representation.

        anchors : Dict[int, AnchorItem]
            Dictionary of pre-processed AnchorItems.

        num_closest_anchors : int
            Number of closest anchors to each sequence to use.

        distance_measure_name : str
            The name of the distance function that will measure the
            distance between each sequence and the anchors.

        distance_measure_kwargs : Dict
            Additional keyword arguments required for the selected
            distance function.

        compression : str, optional
            Compression to use for the output Parquet file.
            Can be 'gzip' or 'snappy', by default 'snappy'.

        chunk_size : int, optional
            Size of DataFrame chunks to process in parallel, by default 5000.
            Larger chunk sizes will consume more memory and result in less
            frequent progress updates, but may result in 
            better performance.

        save_anchor_dists : bool, optional
            Whether to save the binary-encoded distances between a sequence and each anchor
            in the output file, by default False.

        anchor_dist_compression : str, optional
            Compression to use if save_anchor_dists is True , by default 'zlib'.

        anchor_dist_compression_level : int, optional
            Compression level to use if save_anchor_dists is True, by default 9 (highest).

        stop_after_n_chunks : int, optional
            Specify a value larger than 0 if the process should stop after processing
            the given number of chunks, otherwise 0. By default 0.

        nb_workers : int or None, optional
            Number of workers to use for parallel processing. Passing None
            will set to the number of cpus in the system. By default None.
        """

        # Init
        POINT_ID_FIELD = 'sys_point_id'
        FILE_ID_FIELD = 'sys_file_id'
        SEQ_ID_FIELD = 'seq_id'
        base: OASAdapterBase = super(
            OASAdapterJSON, OASAdapterJSON)      # type: ignore

        # Whether json or the newer csv format
        is_miairr_format = fn.endswith('.csv.gz') or fn.endswith('.csv')

        import ast

        # Initialize Pandarallel
        if nb_workers is not None:
            pandarallel.initialize(
                nb_workers=nb_workers,
                use_memory_fs=None,  # auto
                progress_bar=False

            )
        else:
            pandarallel.initialize(
                use_memory_fs=None,  # auto
                progress_bar=False
            )

        # Get arguments to process a single record
        # (for running in parallel)
        process_chunk_args = dict(
            seq_field=seq_field,
            anchors=anchors,
            num_closest_anchors=num_closest_anchors,
            distance_measure_name=distance_measure_name,
            distance_measure_kwargs=distance_measure_kwargs,
            anchor_dist_compression=anchor_dist_compression,
            anchor_dist_compression_level=anchor_dist_compression_level,
            save_anchor_dists=save_anchor_dists
        )

        # Open the file
        processed_count = 0
        with openfunc(fn) as f:

            # Init progress bar
            pbar = tqdm(
                total=record_count, desc="Writing data unit to db...", position=0, leave=True)

            # Skip meta row
            next(f)

            # Start reading in chunks...
            with pd.read_json(f,
                              orient='records',
                              lines=True,
                              chunksize=chunk_size) \
                if not is_miairr_format \
                else pd.read_csv(
                f,
                index_col=None,
                header='infer',
                sep=',',
                error_bad_lines=True,
                chunksize=chunk_size
            ) as reader:

                pqwriter: pq.ParquetWriter = None
                for i_chunk, df_chunk in enumerate(reader):

                    # Stop if required
                    if stop_after_n_chunks > 0 and i_chunk >= stop_after_n_chunks:
                        break

                    # Update status
                    chunk_size = len(df_chunk.index)
                    pbar.update(chunk_size)

                    # Transforms
                    # TODO: Move to OAS_adapter_base?
                    df_chunk = df_chunk.convert_dtypes()

                    # Generate sequential ID (0-based) or use existing field
                    if seq_id_field is None:
                        df_chunk.insert(0, SEQ_ID_FIELD, range(
                            processed_count, processed_count + chunk_size)),
                    else:
                        df_chunk.insert(0, SEQ_ID_FIELD,
                                        df_chunk[seq_id_field])

                    # Insert file id
                    df_chunk.insert(0, FILE_ID_FIELD, file_id)

                    # Create point ID, unique across environment.
                    # Combines file_id and the file-level seq_id.
                    # (May change due to file_id, don't use outside of system)
                    df_chunk.insert(
                        0,
                        POINT_ID_FIELD,
                        df_chunk.apply(
                            lambda row: pointid.create_point_id(
                                row[FILE_ID_FIELD],
                                row[SEQ_ID_FIELD]
                            ),
                            axis=1
                        )
                    )

                    # Compute distances and coordinates
                    # df_computed will be just the computed columns
                    # df_computed = df_chunk.apply(
                    df_computed = df_chunk.parallel_apply(
                        base.process_single_record,
                        axis='columns',
                        result_type='expand',
                        **process_chunk_args
                    )

                    # Add the computed columns
                    df_chunk = pd.concat(
                        [df_chunk, df_computed],
                        axis='columns'
                    )

                    # Convert to PyArrow table
                    pqtable_chunk = pa.Table.from_pandas(df_chunk)

                    # Create Parquet file
                    if i_chunk == 0:
                        pqwriter = pq.ParquetWriter(
                            fn_out,
                            pqtable_chunk.schema,
                            compression=compression
                        )

                    # Append chunk to Parquet file
                    pqwriter.write_table(pqtable_chunk)

                    # Keep count
                    processed_count += chunk_size

            # Close the parquet writer
            if pqwriter:
                pqwriter.close()

            # Finished
            pbar.set_description_str(
                desc='Finished writing to db.', refresh=True)
            pbar.close()