# This script adds the gapped sequence properties to the Parquet record files.

# %% Imports
import argparse
import os
import pandas as pd
from typing import Any, Dict, List

import airrmap.preprocessing.gapseq as gapseq
import airrmap.util.add_parquet_property as addprop
from airrmap.application.config import AppConfig, SeqFileType


def get_gapped_seqs(row: Any,
                    gapped_seq_fields: List[Dict[str, Any]]):
    """
    Creates gapped sequences.

    Parameters
    ----------
    row : Any
        Single row from Pandas DataFrame.

    gapped_seq_fields : List[Dict[str, Any]]
        A list of dictionaries with the following
        properties:
        source_field: The column in the Pandas DataFrame
            containing the original ungapped sequence.
        gapped_field: The new column that should contain
            the created gapped sequence.
        fixed_length: The fixed length of the gapped sequence.

    Returns
    -------
    Dict
        The gapped sequences that will be added to the
        row of the Pandas DataFrame.
    """

    # Init
    result = {}

    # Create gapped sequences
    for gap_field in gapped_seq_fields:
        source_field = gap_field['source_field']
        gapped_field = gap_field['gapped_field']
        fixed_length = gap_field['fixed_length']
        seq = row[source_field]
        
        # Handle NA Type
        seq = '' if isinstance(seq, pd._libs.missing.NAType) else seq

        gapped_seq = gapseq.get_gapped_seq(
            seq=seq,
            fixed_length=fixed_length
        )
        result[gapped_field] = gapped_seq

    # Return
    return result


def main(argv):

    # Process args
    parser = argparse.ArgumentParser(
        description='Add gapped sequences to Parquet record files.'
    )
    parser.add_argument(
        'envfolder',
        type=str,
        help='Environment folder containing envconfig.yaml'
    )
    args = parser.parse_args(argv)
    env_folder = args.envfolder

    # Get env name from folder name
    env_name = os.path.basename(os.path.normpath(env_folder))

    # Load envconfig.yaml
    appcfg = AppConfig()
    envcfg = appcfg.get_env_config(env_name)
    cfgapp = envcfg['application']

    # Fields to create gapped sequences for
    gapped_seq_fields = cfgapp['seq_logos']

    # Output folder
    output_subfolder_name = 'modified'

    # %% Add properties
    addprop.run_add_parquet_property(
        env_name=env_name,
        file_type=SeqFileType.RECORD,
        transform_func=get_gapped_seqs,
        transform_kwargs=dict(
            gapped_seq_fields=gapped_seq_fields
        ),
        output_subfolder_name=output_subfolder_name,
        compression=None,  # use env setting
        chunk_size=None  # use env setting
    )

    # Show completed
    print("Completed, remember to rename folder 'records_modified' to 'records' and refresh the index.")


# %% Main
if __name__ == '__main__':
    #main(None)
    main(['/airrmap-data/GUPTA_CDRH1H2_MultiField'])
