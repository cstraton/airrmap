# Base class containing common functionality
# for processing the study sequences and
# computing the coordinates using anchors and multilateration.
# Can be used to build adapters for other data sources
# (see oas_adapter_json.py).

# tests -> test_oas_adapter_base.py (wip)

# %% Imports
import hashlib
import pathlib
import pandas as pd
import numpy as np
import construct
import airrmap.preprocessing.distance as distance
import operator
import sqlite3
import sqlescapy
import zlib
import json
import ast
import gzip
import sys
import os
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Sequence, Tuple, Any

import airrmap.preprocessing.multilateration as multilateration

# %%


class AnchorItem():
    """
    Represents a single anchor sequence item
    """

    def __init__(self, anchor_id: str, seq: Any,
                 x: float, y: float):
        """
        Create a new AnchorItem instance.

        Parameters
        ----------
        anchor_id : str
            Unique ID of the anchor.

        seq : Any
            The Sequence representation.

        x : float
            The computed x coordinate.

        y : float
            The computed y coordinate.
        """

        self.anchor_id = anchor_id
        self.seq = seq
        self.x = x
        self.y = y


class OASAdapterBase:
    """
    Base class for OAS file adapters.

    Adapter classes derived from this should
    implement the following methods:
    - write_records_file()
    - write_meta_file()
    """

    @staticmethod
    def write_records_file(in_file, out_file):
        raise NotImplementedError()

    @staticmethod
    def write_meta_file(in_file, out_file):
        raise NotImplementedError()

    @staticmethod
    def _sha1file(fn: str) -> str:
        """Get sha1 hash of a file

        Uses buffering to handle large files.

        Args:
            fn (str): File to hash

        Returns:
            (str): Sha1 hash of file
        """

        # Adapted from:
        # https://stackoverflow.com/questions/22058048/hashing-a-file-in-python

        BUFFER_SIZE = 65536
        sha1 = hashlib.sha1()

        with open(fn, 'rb') as f:
            while True:
                data = f.read(BUFFER_SIZE)
                if not data:
                    break
                sha1.update(data)

        return sha1.hexdigest()

    @staticmethod
    def get_extra_file_meta(file_id, fn, record_count: int) -> Dict:
        """
        Get meta properties for a given file.

        Parameters
        ----------
        file_id: int
            The numeric file id. Needs to match to file_id in the the records file.

        fn : file or file-like
            The file to get properties, typically a file path.

        record_count : int
            The number of records in the file (excluding the header).

        Returns
        -------
        Dict
            The populated dictionary
        """
        return dict(
            sys_file_id=file_id,
            sys_filename=pathlib.Path(fn).name,
            sys_data_unit_sha1=OASAdapterBase._sha1file(fn),
            sys_processed_utc=datetime.now(timezone.utc).isoformat() + 'Z',
            sys_records=record_count,
            sys_file_size=os.path.getsize(fn),
            sys_endian=sys.byteorder
        )

    @staticmethod
    def header_to_df(header: Dict[str, Any],
                     extra: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Convert header to tall Pandas DataFrame

        Parameters
        ----------
        header : Dict[str, Any]
            The original file header with file-level properties.

        extra : Dict[str, Any], optional
            Dictionary of additional properties.
            Will be added at the beginning of the returned DataFrame.
            By default None.

        Returns
        -------
        pd.DataFrame
            Tall DataFrame containing property_name | property_value
        """

        # Convert header line to format:
        # property_name | property_value
        # Longitudinal  | no
        # Chain         | Heavy
        # ...
        if extra is not None:
            records = [item for item in extra.items()]
        else:
            records = []

        records.extend(
            [item for item in header.items()]
        )

        df = pd.DataFrame.from_records(records)
        df.columns = ('property_name', 'property_value')
        return df

    @staticmethod
    def get_distances(item1: Any,
                      items: Dict[int, AnchorItem],
                      measure_function: Callable,
                      **kwargs) -> OrderedDict:
        """Measures distance between one item and a number of other items.

        TODO: Optimize
        Items should be of the type required for the measure_function.

        Args:
            item1: The base item.
            items (dict[int, AnchorItem]): Dictionary of other items to measure distance to.
            measure_function (function): Distance measure function.
            **kwargs: Additional args for the measure_function (optional).

        Returns:
            OrderedDict[int, Any]: List of ids and associated distances, in order of distance.
        """

        distances = {}

        # Get distance from seq to each anchor
        for item2key, item2 in items.items():
            dist = measure_function(item1, item2.seq, **kwargs)
            distances[item2key] = dist

        # Sort in order of ascendng distance
        ordered_distances: OrderedDict = OrderedDict(
            sorted(
                distances.items(),
                key=operator.itemgetter(1),
                reverse=False
            )
        )

        # Return
        return ordered_distances

    @staticmethod
    def load_anchors(fn_anchors: str,
                     seq_column: str = 'aa_annotated',
                     seq_transform: Callable[[Any], Any] = None,
                     orderby_column: str = 'anchor_name',
                     ) -> Dict[int, AnchorItem]:
        """
        Load anchors sequences.

        Format of sequences should match those that will be compared
        with OAS.

        NOTE! seq_column and orderby_column should not be provided
        by unauthorised users as potential for sql injection. Currently appear
        unable to provide field names as parameterised query for SQLite.

        Parameters
        ----------
        fn_anchors : str
             Db containing the translated anchor records (reads from
             anchor_records table).

        seq_column : str, optional
            Column containing the representation of the sequence, by default 'aa_annotated'.

        seq_transform: Callable[Any], optional
            A function that transforms the value in seq_column, by default None.

        orderby_column : str, optional
            Column to order by, by default 'anchor_name'.

        Returns
        -------
        Dict[int, AnchorItem]
            Dictionary of anchors, key=[id_column], value=[AnchorItem]

        Example
        -------
        ```
        load_anchors('anchors.db')
        > {'anchor1': AnchorItem(anchor_id=1, seq={'cdrh1': {'1': 'A' ...}}, x=0.12, y=1.23)}
        ```
        """

        # Init
        seq_column = sqlescapy.sqlescape(seq_column)
        orderby_column = sqlescapy.sqlescape(orderby_column)

        sql = f"""
            SELECT rec.anchor_id AS anchor_id,
                   rec.{seq_column} AS seq,
                   coord.x AS x,
                   coord.y AS y
            FROM anchor_records AS rec
            LEFT JOIN anchor_coords AS coord
            ON rec.anchor_id = coord.anchor_id
            ORDER BY rec.{orderby_column}
        """

        # Load anchor sequences
        with sqlite3.connect(fn_anchors) as conn:
            df_anchors = pd.read_sql(
                sql=sql,
                con=conn
            )

        # Transform and place in dictionary (AnchorItem)
        transform_fn: Callable = \
            seq_transform if seq_transform is not None else lambda x: x

        anchors = {
            anchor_id: AnchorItem(
                anchor_id=anchor_id,
                seq=transform_fn(seq),
                x=x,
                y=y
            )
            for anchor_id, seq, x, y in
            zip(list(df_anchors['anchor_id']),
                list(df_anchors['seq']),
                list(df_anchors['x']),
                list(df_anchors['y'])
                )
        }

        # Return
        return anchors

    @staticmethod
    def anchor_dist_codec(size: int) -> construct.Struct:
        """Define anchor_dists binary coder/decoder

        VarInt varies size of integer in bytes.
        Float32n is native endianness 32-bit float (endianness of the encoding
        machine, if required, will be stored in the meta table of the same db).
        dist_checksum will contain int(sum(anchor_dists)) and can be used to verify
        the distances have been decoded correctly.

        See https://construct.readthedocs.io/en/latest/advanced.html#bytes-and-bits

        Error will occur if trying to encode incorrect number of items
        (as defined by size argument).
        """
        anchor_dist_codec = construct.Struct(
            "signature" / construct.Const(b"DST"),
            "dist_checksum" / construct.VarInt,
            "anchor_ids" / construct.Array(size, construct.VarInt),
            "anchor_dists" / construct.Array(size, construct.Float32n)
        )

        return anchor_dist_codec

    @staticmethod
    def anchor_dist_encode(anchor_dists: OrderedDict,
                           codec: construct.Struct,
                           compression: str = 'zlib',
                           compression_level: int = 9) -> bytes:
        """Binary encodes anchor distances for BLOB field storage.

        The distance from a sequence to each anchor must be computed for
        the distance matrix, resulting in large data sizes.
        For example, 357,000 OAS sequences * 300 anchor sequences =
        ~107 million records, for just one Data Unit. Therefore, we
        use binary encoding and compression to reduce the size.
        The resulting bytes can be stored in a db BLOB field.

        Args:
            anchor_dists (OrderedDict[int, Any]): Ordered dictionary
                where key=anchor_id value=distance. Should be in
                ascending order of distance.

            codec (construct.Struct): Encoder/Decoder definition.

            compression (str, optional): Compression method, 'zlib' or
                'gzip' or ''. Defaults to 'zlib'.

            compression_level (int, optional): Level of compression
                for gzip or zlib. Defaults to 9.

        Returns:
            bytes: The encoded bytes.
        """

        # Number of items
        size = len(anchor_dists)

        # Encode
        dist_bytes = codec.build(
            dict(
                dist_checksum=int(sum(anchor_dists.values())),
                anchor_ids=list(anchor_dists.keys()),
                anchor_dists=list(anchor_dists.values())
            )
        )

        # Compress
        if compression == 'zlib':
            dist_bytes = zlib.compress(dist_bytes, level=compression_level)
        elif compression == 'gzip':
            dist_bytes = gzip.compress(
                dist_bytes, compresslevel=compression_level)
        elif compression is None or compression == '':
            pass
        else:
            raise Exception(f'Unexpected compression method: {compression}')

        # Return
        return dist_bytes

    @staticmethod
    def anchor_dist_decode(encoded_bytes: bytes,
                           codec: construct.Struct,
                           compression: str = 'zlib') -> construct.Container:
        """Decodes anchor distances from bytes

        Args:
            encoded_bytes (bytes): The encoded (and potentially compressed) bytes.
            codec (construct.Struct): Construct codec for encoding/decoding.
            compression (str, optional): 'zlib', 'gzip' or ''. Defaults to 'zlib'.

        Raises:
            Exception: If invalid compression level is supplied.

        Returns:
            construct.Container: The decoded bytes.
        """

        # Decompress
        if compression == 'zlib':
            byts = zlib.decompress(encoded_bytes)
        elif compression == 'gzip':
            byts = gzip.decompress(encoded_bytes)
        elif compression is None or compression == '':
            byts = encoded_bytes
        else:
            raise Exception(f'Unexpected compression method: {compression}')

        # Return decoded
        return codec.parse(byts)

    @staticmethod
    def compute_sequence_coords(anchor_dists: OrderedDict,
                                anchors: Dict[int, AnchorItem],
                                num_closest_anchors: int = 0,
                                use_mae: bool = False) -> Dict[str, Any]:
        """
        Compute the xy coordinates for a single sequence.

        NOTE! anchor_dists should be OrderedDict in order of distance,
        which will apply if num_closest_anchors is used.

        This process takes the known anchor coordinates and distances
        to each anchor, and attempts to find the optimal position for
        the sequence.

        Parameters
        ----------
        anchor_dists : OrderedDict[int, Any]
            Ordered dictionary of anchor_ids and the numeric distance from
            sequence to each anchor. -Should be in order of ascending distance-.

        anchors : Dict[int, AnchorItem]
            [description]

        num_closest_anchors : int, optional
            Number of closest anchors to use. Use negative value
            to get n furthest away. By default 0 (all).

        use_mae : bool, optional
            True to use mean absolute error instead of mean squared error
            when computing the coordinates. By default False.

        Returns
        -------
        Dict[str, Any]
            Returns a record containing the x, y coordinates
            and additional metadata.
        """

        # NOTE! anchor_dists should be OrderedDict in ascending order.
        # ^^Perf: 158 ns

        # Check if closest_anchors specified (0 to use all)
        # != 0 instead of > 0, in case we want the ones furthest away (could use -10)
        use_closest_anchors = num_closest_anchors != 0

        # Get list of n closest anchor IDs (or all)
        closest_anchor_ids = list(anchor_dists.keys()) if not use_closest_anchors \
            else list(anchor_dists.keys())[:num_closest_anchors]

        # Get list of n closest anchor items (or all)
        closest_anchor_items = list(anchors.values()) if not use_closest_anchors \
            else [anchors[anchor_id]
                  for anchor_id in closest_anchor_ids]

        # Check we have everything
        if len(closest_anchor_ids) != len(closest_anchor_items):
            raise Exception('Not all anchor ids are accounted for.')

        # ^^ Perf: 8.4 µs

        # Get closest anchor items, build values
        #closest_anchor_coords = np.zeros()
        closest_anchor_coords = []
        closest_anchor_dists = []
        x_sum = 0.
        y_sum = 0.
        for anchor_id in closest_anchor_ids:
            anchor_item = anchors[anchor_id]
            x_sum += anchor_item.x
            y_sum += anchor_item.y
            closest_anchor_coords.append((anchor_item.x, anchor_item.y))
            closest_anchor_dists.append(anchor_dists[anchor_id])

        #  ^^ Perf: 103 µs

        # Compute the midpoint
        num_anchors_used = num_closest_anchors if num_closest_anchors != 0 else len(
            closest_anchor_ids)
        x_mid = x_sum / num_anchors_used
        y_mid = y_sum / num_anchors_used
        anchor_midpoint_coord = tuple((x_mid, y_mid))

        #  ^^ Perf: 103 µs

        # Convert to ndarray - otherwise will be very slow!
        # (For some reason, ~78x faster passing ndarray instead of list to minimize() function)
        a_closest_anchor_coords = np.array(closest_anchor_coords)
        a_closest_anchor_dists = np.array(closest_anchor_dists)

        # Compute coordinates for sequence
        seq_coords = multilateration.calculate_coords(
            initial_coords=anchor_midpoint_coord,
            anchor_coords=a_closest_anchor_coords,
            anchor_distances=a_closest_anchor_dists,
            use_mae=use_mae)

        # Get result
        x, y = seq_coords.x
        if not seq_coords.success:
            raise Exception(
                f'Error during calculation of coordinates: {seq_coords.message}')

        # Return results
        record_out = {
            'x': float(x),
            'y': float(y),
            'init_x': float(anchor_midpoint_coord[0]),
            'init_y': float(anchor_midpoint_coord[1]),
            'num_closest_anchors': num_closest_anchors
        }

        return record_out

        #  ^^ Perf: 1.92 ms

    @staticmethod
    def prepare(fn, seq_row_start: int, fn_anchors,
                anchor_seq_field='aa_annotated'):
        """
        Prepare for processing a file.

        Enables some arguments to be computed only once
        as required for both the meta and records file.

        Parameters
        ----------
        fn : str, file-like or None
            The data unit to process. Pass None if using
            this function directly / without a Data Unit file
            (for loading the anchor sequences for example).

        seq_row_start : int
            The 1-based row that the sequence records start at
            (e.g. 2 for json files, 3 for csv).

        fn_anchors : str or file-like
            The database/file containing the processed anchors.

        anchor_seq_field: str
            Field in the anchor file containing the sequence to be
            measured.

        Returns
        -------
        Dict
            Initial args required for processing of the data unit.

            record_count: Number of records in the data unit file, or
                zero if 'fn' is None.

            openfunc: The open function to read the file with (e.g. gzip.open)

            anchors: Dictionary of AnchorItems loaded from the anchor db.
                Key=anchor_id, Value=AnchorItem().

            anchors_file_sha1: SHA-1 of the anchors fn_anchors file.
        """

        JSON_SINGLE_QUOTE = 2

        # Open function depending on .gz or .json file
        openfunc: Any = gzip.open if str(fn).endswith('.gz') else open

        # Get line count (minus headers).
        # Return 0 if not using a data unit file.
        if fn is not None:
            record_count = sum(1 for line in openfunc(
                fn, 'rt')) - (seq_row_start - 1)
        else:
            record_count = 0

        # Load anchors
        anchors: Dict[int, AnchorItem] = OASAdapterBase.load_anchors(
            fn_anchors=fn_anchors,
            seq_column=anchor_seq_field,
            seq_transform=None
        )

        # Get sha1 of the anchors daabase
        anchors_file_sha1 = OASAdapterBase._sha1file(fn_anchors)

        # Return config
        return dict(
            record_count=record_count,
            openfunc=openfunc,
            anchors=anchors,
            anchors_file_sha1=anchors_file_sha1
        )

    @staticmethod
    def load_seq_value(x):
        """
        Load a sequence value direct from the dataset and
        handle any required transformations (based on type, for example).

        Parameters
        ----------
        x : Any
            Sequence value(s) from dataset.

        Returns
        -------
        Any
            Transformed seq (if applicable).
        """

        return x

    @staticmethod
    def process_single_record(row,
                              seq_field: str,
                              anchors: Dict[int, AnchorItem],
                              num_closest_anchors: int,
                              distance_measure_name: str,
                              distance_measure_kwargs: dict,
                              save_anchor_dists: bool = True,
                              anchor_dist_compression: str = 'zlib',
                              anchor_dist_compression_level: int = 9,
                              use_mae: bool = False) -> Dict[str, Any]:
        """
        Process a single sequence record from a DataFrame row (series) or dict-like item.

        Computes the distance to each anchor, then computes the coordinates of the sequence.


        Parameters
        ----------
        row : Series or dict-like
            The record containing the sequence.

        seq_field : str
            The field in the row containing the sequence representation.

        anchors : Dict[int, AnchorItem]
            Dictionary of AnchorItem instances. Key=anchor_id (int).

        num_closest_anchors : int
            Number of closest anchors to use for computation of the
            sequence coordinates. If 0 then use all provided anchors.

        distance_measure_name : str
            Name of the distance measure to use.

        distance_measure_kwargs : dict
            Additional keyword arguments required by the 
            selected distance measure.

        save_anchor_dists : bool, optional
            True to include the binary-encoded anchor distances in the
            result, otherwise False. By default True.

        anchor_dist_compression : str, optional
            The compression type to use for the binary-encoded 
            anchor distance ('gzip' or 'zlib'). By default 'zlib'.

        anchor_dist_compression_level : int, optional
            Compression level 0-9, 9 is highest. By default 9.

        use_mae : bool, optional
            True to use mean absolute error instead of
            mean squared error when computing sequence coordinates.
            By default False.


        Returns
        -------
        Dict[str, Any]
            The computed sequence coordinates, number of anchors used,
            and optionally the binary-encoded anchor distances (all).
            The keys of this dictionary determine the column names used
            in the processed file.
        """

        # Get distance function from str (functions not
        # supported by pickle; required for multiprocessing)
        distance_measure_fn = getattr(distance, distance_measure_name)

        # Get codec (~14 µs)
        if save_anchor_dists:
            anchor_dist_codec = OASAdapterBase.anchor_dist_codec(
                size=len(anchors))

        # Get (and transform) the sequence from the DataFrame row.
        # Convert from JSON if required.
        seq: Any
        seq = OASAdapterBase.load_seq_value(
            x=row[seq_field]
        )

        # Get distance between sequence and all anchor sequences
        ordered_anchor_dists: OrderedDict[int, Any] = OASAdapterBase.get_distances(
            item1=seq,
            items=anchors,
            measure_function=distance_measure_fn,
            **distance_measure_kwargs
        )

        # Compute the coordinates
        seq_coords: Dict = OASAdapterBase.compute_sequence_coords(
            anchor_dists=ordered_anchor_dists,
            anchors=anchors,
            num_closest_anchors=num_closest_anchors,
            use_mae=use_mae
        )

        # Binary-encode distances and compress
        if save_anchor_dists:
            anchor_dist_bytes = OASAdapterBase.anchor_dist_encode(
                anchor_dists=ordered_anchor_dists,
                codec=anchor_dist_codec,
                compression_level=anchor_dist_compression_level,
                compression=anchor_dist_compression,
            )

        # Create result - keys will determine name of new columns
        # in Pandas DataFrame apply()
        result = dict(
            sys_coords_x=seq_coords['x'],
            sys_coords_y=seq_coords['y'],
            sys_coords_init_x=seq_coords['init_x'],
            sys_coords_init_y=seq_coords['init_y'],
            sys_coords_num_closest_anchors=seq_coords['num_closest_anchors']
        )

        # Add binary encoded anchor distances if required
        if save_anchor_dists:
            result['sys_anchor_dists_anchor_count'] = len(ordered_anchor_dists)
            result['sys_anchor_dists_anchor_dists'] = anchor_dist_bytes

        # Return the result
        return result
# %%
