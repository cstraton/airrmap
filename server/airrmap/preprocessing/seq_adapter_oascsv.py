# Adapter for OAS datasets.
# Calls multiple functions in the base
# class SeqAdapterBase.
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
from airrmap.preprocessing.seq_adapter_base import SeqAdapterBase, AnchorItem


# %%


class SeqAdapterOASCSV(SeqAdapterBase):

    @staticmethod
    def process_meta(file_id: int,
                     fn,
                     fn_out,
                     record_count: int, fn_anchors_sha1: str,
                     anchor_dist_compression,
                     anchor_dist_compression_level):

        FILE_ID_FIELD = 'sys_file_id'

        # Open and read meta
        # REF: http://opig.stats.ox.ac.uk/webapps/oas/documentation
        header_meta = ','.join(pd.read_csv(fn, nrows=0).columns)
        header_dict = json.loads(header_meta)

        # Get additional file properties
        # super(a, a) required for static method
        extra_properties = super(SeqAdapterOASCSV, SeqAdapterOASCSV).get_extra_file_meta(
            file_id=file_id,
            fn=fn,
            record_count=record_count)

        # Get anchor distance params
        extra_properties['sys_dist_compression'] = anchor_dist_compression
        extra_properties['sys_dist_compression_level'] = anchor_dist_compression_level
        extra_properties['sys_dist_anchor_db_sha1'] = fn_anchors_sha1

        # Convert to long DataFrame
        # # super(a, a) required for static method
        df_header = super(SeqAdapterOASCSV, SeqAdapterOASCSV).header_to_df(
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
                        record_count: int,
                        seq_id_field: str,
                        anchors: Dict[int, AnchorItem],
                        num_closest_anchors: int,
                        distance_measure_name: str,
                        distance_measure_env_kwargs: Dict,
                        distance_measure_seq_kwargs: Dict,
                        distance_measure_anchor_kwargs: Dict,
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

        record_count : int
            Number of sequences/records in the data (excluding headers)

        seq_id_field : str or None
            The integer field in the data unit to use as the sequence ID,
            unique within the data unit file (a separate ID will be created
            to make unique within the environment).
            Specify None to auto-generate.

        anchors : Dict[int, AnchorItem]
            Dictionary of pre-processed AnchorItems.

        num_closest_anchors : int
            Number of closest anchors to each sequence to use.

        distance_measure_name : str
            The name of the distance function that will measure the
            distance between each sequence and the anchors.

        distance_measure_env_kwargs : Dict
            Environment-level arguments required for the
            selected distance measure.

        distance_measure_seq_kwargs: Dict
            Record-level arguments for the sequence records,
            required for the selected distance measure.

        distance_measure_anchor_kwargs : Dict
            Record-level arguments for the anchor records,
            required for the selected distance measure.

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
        base: SeqAdapterBase = super(
            SeqAdapterOASCSV, SeqAdapterOASCSV)      # type: ignore

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
            anchors=anchors,
            num_closest_anchors=num_closest_anchors,
            distance_measure_name=distance_measure_name,
            distance_measure_env_kwargs=distance_measure_env_kwargs,
            distance_measure_seq_kwargs=distance_measure_seq_kwargs,
            distance_measure_anchor_kwargs=distance_measure_anchor_kwargs,
            anchor_dist_compression=anchor_dist_compression,
            anchor_dist_compression_level=anchor_dist_compression_level,
            save_anchor_dists=save_anchor_dists
        )

        # Open the file
        processed_count = 0

        # Init progress bar
        pbar = tqdm(
            total=record_count, desc="Writing data unit to db...", position=0, leave=True)

        # Start reading in chunks...
        # OAS reading files: http://opig.stats.ox.ac.uk/webapps/oas/documentation
        with pd.read_csv(
            fn,
            index_col=None,
            header=1,
            sep=',',
            error_bad_lines=True,
            chunksize=chunk_size,
            dtype=SeqAdapterOASCSV.get_csv_dtypes()
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
                # TODO: Move to seq_adapter_base?
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

    @staticmethod
    def get_csv_dtypes() -> Dict:
        """
        Return explicit dtypes for each column in an OAS csv file.

        Added as Pandas raises a mixed-type error if inferred data types
        vary between read chunks.
        See https://stackoverflow.com/questions/24251219/pandas-read-csv-low-memory-and-dtype-options.

        Returns
        -------
        Dict
            The dtype mappings.
        """

        # string[pyarrow] is generally more efficient, available in Pandas >= 1.3.
        # https://pythonspeed.com/articles/pandas-string-dtype-memory/

        # All ints are specified as float32 as some ints in the OAS
        # csv files are floats (e.g. "236.0") which causes an error if trying to convert to uint16:
        # e.g. "ValueError: cannot safely convert passed user dtype of uint16
        #      for float64 dtyped data in column 17".

        # Also float16 is not supported in pyarrow / Parquet:
        # "pyarrow.lib.ArrowNotImplementedError: Unhandled type for Arrow to Parquet schema conversion: halffloat"
        # See: https://github.com/apache/arrow/issues/2691

        # TODO: Look at use of use_byte_stream_split (better compression for floats).
        # REF: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html

        # Column mappings can be added as required - only columns that
        # match will be taken into account by pd.read_csv(dtype={...}).
        # Non-matches will be ignored.
        dtypes = {
            'sequence': 'string[pyarrow]',
            'locus': 'string[pyarrow]',
            'stop_codon': 'string[pyarrow]',
            'vj_in_frame': 'string[pyarrow]',
            'v_frameshift': 'string[pyarrow]',
            'productive': 'string[pyarrow]',
            'rev_comp': 'string[pyarrow]',
            'complete_vdj': 'string[pyarrow]',
            'v_call': 'string[pyarrow]',
            'd_call': 'string[pyarrow]',
            'j_call': 'string[pyarrow]',
            'sequence_alignment': 'string[pyarrow]',
            'germline_alignment': 'string[pyarrow]',
            'sequence_alignment_aa': 'string[pyarrow]',
            'germline_alignment_aa': 'string[pyarrow]',
            'v_alignment_start': 'float32',
            'v_alignment_end': 'float32',
            'd_alignment_start': 'float32',
            'd_alignment_end': 'float32',
            'j_alignment_start': 'float32',
            'j_alignment_end': 'float32',
            'v_sequence_alignment': 'string[pyarrow]',
            'v_sequence_alignment_aa': 'string[pyarrow]',
            'v_germline_alignment': 'string[pyarrow]',
            'v_germline_alignment_aa': 'string[pyarrow]',
            'd_sequence_alignment': 'string[pyarrow]',
            'd_sequence_alignment_aa': 'string[pyarrow]',
            'd_germline_alignment': 'string[pyarrow]',
            'd_germline_alignment_aa': 'string[pyarrow]',
            'j_sequence_alignment': 'string[pyarrow]',
            'j_sequence_alignment_aa': 'string[pyarrow]',
            'j_germline_alignment': 'string[pyarrow]',
            'j_germline_alignment_aa': 'string[pyarrow]',
            'fwr1': 'string[pyarrow]',
            'fwr1_aa': 'string[pyarrow]',
            'cdr1': 'string[pyarrow]',
            'cdr1_aa': 'string[pyarrow]',
            'fwr2': 'string[pyarrow]',
            'fwr2_aa': 'string[pyarrow]',
            'cdr2': 'string[pyarrow]',
            'cdr2_aa': 'string[pyarrow]',
            'fwr3': 'string[pyarrow]',
            'fwr3_aa': 'string[pyarrow]',
            'fwr4': 'string[pyarrow]',
            'fwr4_aa': 'string[pyarrow]',
            'cdr3': 'string[pyarrow]',
            'cdr3_aa': 'string[pyarrow]',
            'junction': 'string[pyarrow]',
            'junction_length': 'float32',
            'junction_aa': 'string[pyarrow]',
            'junction_aa_length': 'float32',
            'v_score': 'float32',
            'd_score': 'float32',
            'j_score': 'float32',
            'v_cigar': 'string[pyarrow]',
            'd_cigar': 'string[pyarrow]',
            'j_cigar': 'string[pyarrow]',
            'v_support': 'float32',
            'd_support': 'float32',
            'j_support': 'float32',
            'v_identity': 'float32',
            'd_identity': 'float32',
            'j_identity': 'float32',
            'v_sequence_start': 'float32',
            'v_sequence_end': 'float32',
            'v_germline_start': 'float32',
            'v_germline_end': 'float32',
            'd_sequence_start': 'float32',
            'd_sequence_end': 'float32',
            'd_germline_start': 'float32',
            'd_germline_end': 'float32',
            'j_sequence_start': 'float32',
            'j_sequence_end': 'float32',
            'j_germline_start': 'float32',
            'j_germline_end': 'float32',
            'fwr1_start': 'float32',
            'fwr1_end': 'float32',
            'cdr1_start': 'float32',
            'cdr1_end': 'float32',
            'fwr2_start': 'float32',
            'fwr2_end': 'float32',
            'cdr2_start': 'float32',
            'cdr2_end': 'float32',
            'fwr3_start': 'float32',
            'fwr3_end': 'float32',
            'fwr4_start': 'float32',
            'fwr4_end': 'float32',
            'cdr3_start': 'float32',
            'cdr3_end': 'float32',
            'np1': 'string[pyarrow]',
            'np1_length': 'float32',
            'np2': 'string[pyarrow]',
            'np2_length': 'float32',
            'c_region': 'string[pyarrow]',
            'Redundancy': 'float32',
            'ANARCI_numbering': 'string[pyarrow]',
            'ANARCI_status': 'string[pyarrow]',
        }
        return dtypes
