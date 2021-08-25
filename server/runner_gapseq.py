# This script adds the gapped sequence properties to the Parquet record files.
# Note that modified files will be added to ./modified subfolder first.

# %% Imports
import argparse
import yaml
import os
from typing import Union, Dict

import airrmap.preprocessing.imgt as imgt
import airrmap.util.add_parquet_property as addprop
from airrmap.application.config import AppConfig, SeqFileType
from airrmap.preprocessing.oas_adapter_base import OASAdapterBase

# %%


def get_imgt_gapped(row,
                    seq_field,
                    convert_json: Union[bool, int],
                    imgt_residue_names: Dict,
                    imgt_region_ranges: Dict):
    """
    Read numbered sequence and convert to a gapped string.

    Parameters
    ----------
    row : Any
        Single row from Pandas DataFrame.

    seq_field : Any
        The column containing the numbered sequence value.

    convert_json : Union[bool, int],
        JSON type, TRUE/1 for double quoted, or 2 for single quoted.

    imgt_residue_names : Dict
        Dictionary of residue names, where key is the IMGT/OAS-numbered
        residue and value is the zero-based index in the gapped string.

    imgt_region_ranges : Dict
        Dictionary, where keys are region names (e.g. 'cdr1'), and values
        are tuples containing (from, to) positions (1-based, inclusive).
        Note, chain (e.g. 'h') should not be included in the region name.

    Returns
    -------
    str
        The gapped sequence string.
    """

    # Init
    result = {}

    # Load the sequence
    seq = OASAdapterBase.load_seq_value(
        x=row[seq_field],
        convert_json=convert_json
    )

    # Convert to gapped
    gapped_seq = imgt.imgt_to_gapped(
        imgt_seq=seq,
        imgt_residue_names=imgt_residue_names
    )
    result['seq_gapped'] = gapped_seq

    # Add individual regions
    for region_name in imgt_region_ranges.keys():
        from_pos, to_pos = imgt_region_ranges[region_name]
        # +1 as ranges are 0-based inclusive and slice is exclusive.
        region_seq = gapped_seq[from_pos: to_pos + 1]
        result[f'seq_gapped_{region_name}'] = region_seq

    # Return, seq_gapped, seq_gapped_fw1 etc. will be the new fields.
    return result


def main(argv):

    # Process args
    parser = argparse.ArgumentParser(
        description='Add gapped sequences to Parquet record files.')
    parser.add_argument('envfolder', type=str,
                        help='Environment folder containing envconfig.yaml')

    args = parser.parse_args(argv)
    env_folder = args.envfolder

    # Get env name from folder name
    env_name = os.path.basename(os.path.normpath(env_folder))

    # Load envconfig.yaml
    appcfg = AppConfig()
    envcfg = appcfg.get_env_config(env_name)
    cfgseq = envcfg['sequence']
    cfganc = envcfg['anchor']

    # %% init
    numbered_seq_field = envcfg['application']['numbered_seq_field']
    # Default to true unless specified (will error if incorrect value)
    if 'numbered_seq_field_is_json' in envcfg['application']:
        numbered_seq_field_is_json = envcfg['application']['numbered_seq_field_is_json']
    else:
        numbered_seq_field_is_json = True
    output_subfolder_name = 'modified'
    imgt_residue_names = imgt.get_imgt_residue_names(as_dict=True)

    # Get regions
    # Chain doesn't matter, we want 'cdr1' instead of 'cdrh1' for consistency.
    imgt_region_ranges = imgt.get_region_positions(
        chain='',
        as_dict=True,
        is_rearranged=True,
        for_gapped_seq=True
    )

    # %% Add properties
    addprop.run_add_parquet_property(
        env_name=env_name,
        file_type=SeqFileType.RECORD,
        transform_func=get_imgt_gapped,
        transform_kwargs=dict(
            seq_field=numbered_seq_field,
            convert_json=numbered_seq_field_is_json,
            imgt_residue_names=imgt_residue_names,
            imgt_region_ranges=imgt_region_ranges
        ),
        output_subfolder_name=output_subfolder_name,
        compression=None,  # use env setting
        chunk_size=None  # use env setting
    )

    # Show completed
    print("Completed, remember to rename folder 'records_modified' to 'records' and refresh the index.")


# %% Main
if __name__ == '__main__':
    main(None)
