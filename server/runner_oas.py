# Runner to process the study sequences / data units.
# Creates 'file_list.json' which will have
# a list of all files being processed, along with an ID and
# status. If the process fails halfway through, it will skip
# any files already processed on the next run.


# %% Imports
import yaml
from pathlib import Path
import helpers
import logging
import os
import json
import glob
import argparse
from airrmap.preprocessing.oas_adapter_json import OASAdapterJSON
from airrmap.preprocessing.oas_adapter_base import OASAdapterBase

from tqdm import tqdm
from typing import Any, Dict, List


# File states
class FileStatus:
    ready = 'READY'
    completed = 'COMPLETED'
    processing = 'PROCESSING'


# %%  Module to run all


def main(argv):

    # Turn off numba DEBUG messages
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)

    # Process args
    parser = argparse.ArgumentParser(
        description='Build seq db for data units.')
    parser.add_argument('envfolder', type=str,
                        help='Environment folder containing envconfig.yaml')

    args = parser.parse_args(argv)
    env_folder = args.envfolder

    # Load envconfig.yaml
    fn_config = os.path.join(env_folder, 'envconfig.yaml')
    with open(fn_config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfgseq = cfg['sequence']
    cfganc = cfg['anchor']

    # Get params
    distance_measure = cfg['distance_measure']
    distance_measure_kwargs = dict(regions=cfganc['regions'])
    fn_anchor_db = os.path.join(
        env_folder, cfganc['build_folder'], cfganc['build_db_file'])
    anchor_seq_field = cfganc['seq_field']
    anchor_seq_field_is_json = cfganc['seq_field_is_json']
    num_closest_anchors = cfgseq['num_closest_anchors']
    save_anchor_dists = cfgseq['save_anchor_dists']
    anchor_dist_compression = cfgseq['anchor_dist_compression']
    anchor_dist_compression_level = cfgseq['anchor_dist_compression_level']
    seq_field = cfgseq['seq_field']
    seq_field_is_json = cfgseq['seq_field_is_json']
    seq_id_field = cfgseq['seq_id_field']
    seq_row_start = cfgseq['seq_row_start']
    output_file_compression = cfgseq['output_file_compression']
    process_chunk_size = cfgseq['process_chunk_size']
    process_nb_workers = cfgseq['process_nb_workers']

    # Create necessary folders
    # TODO: Allow records and meta folders to be configurable (also check for other code)
    os.makedirs(os.path.join(
        env_folder, cfgseq['build_folder']), exist_ok=True)
    os.makedirs(os.path.join(
        env_folder, cfgseq['build_image_folder']), exist_ok=True)
    os.makedirs(os.path.join(
        env_folder, cfgseq['build_folder'], 'meta'), exist_ok=True)
    os.makedirs(os.path.join(
        env_folder, cfgseq['build_folder'], 'records'), exist_ok=True)

    # Path for list of files
    fn_file_list = os.path.join(
        env_folder,
        cfgseq['build_folder'],
        'file_list.json'
    )

    # Read file list and sys_file_id values
    # if exists, otherwise create new
    if os.path.isfile(fn_file_list):
        file_list_existing = True
        with open(fn_file_list, 'r') as f:
            files_to_process = json.load(f)
    else:
        # Get list of files to process (returns full path)
        file_list_existing = False
        files_to_process: List[Dict[str, Any]] = get_files_to_process(
            os.path.join(env_folder, cfgseq['src_folder']),
            cfgseq['src_file_pattern']
        )

    # Get logger
    log = helpers.init_logger('applog',
                              os.path.join(env_folder, cfgseq['build_folder'], 'log.txt'))

    # Log start with passed in config
    log.info('---- Start Process OAS procedure. ----')
    for k, v in locals().items():
        log.info(f'{k}: {v}')

    # Process each file
    skipped_files = 0
    log.info('-- Starting main loop... --')
    for data_unit_item in tqdm(files_to_process, desc='Processing data units...'):

        # Data unit
        file_id = data_unit_item['file_id']
        fn_data_unit = data_unit_item['file_path']
        data_unit_status = data_unit_item['status']
        data_unit_name = Path(fn_data_unit).name

        log.info(
            f'PROCESSING {Path(fn_data_unit).name}, status is {data_unit_status}...')

        # Skip if already processed
        if data_unit_status == FileStatus.completed:
            skipped_files += 1
            log.info(
                f'SKIPPING {Path(fn_data_unit).name} as status is {data_unit_status}.')
            continue

        # Get output meta filename
        fn_data_unit_meta = os.path.join(
            env_folder,
            cfgseq['build_folder'],
            'meta',
            data_unit_name + '.meta.parquet'
        )

        # Get output records filename
        fn_data_unit_record = os.path.join(
            env_folder,
            cfgseq['build_folder'],
            'records',
            data_unit_name + '.parquet'
        )

        # Prepare some of the args
        # such as record count
        log.info('Preparing initial arguments...')
        prep_args: Dict = OASAdapterBase.prepare(
            fn=fn_data_unit,
            seq_row_start=seq_row_start,
            fn_anchors=fn_anchor_db,
            anchor_seq_field=anchor_seq_field,
            anchor_convert_json=anchor_seq_field_is_json
        )
        log.info('Finished preparing initial arguments.')

        # TODO: CLear down meta and records Parquet files first
        # and create the folders.

        # Write the meta file
        log.info('Start writing meta file...')
        OASAdapterJSON.process_meta(
            file_id=file_id,
            fn=fn_data_unit,
            fn_out=fn_data_unit_meta,
            openfunc=prep_args['openfunc'],
            record_count=prep_args['record_count'],
            fn_anchors_sha1=prep_args['anchors_file_sha1'],
            anchor_dist_compression=anchor_dist_compression,
            anchor_dist_compression_level=anchor_dist_compression_level
        )
        log.info('Finished writing meta file.')

        # Compute distances and coordinates, and
        # write the records file.
        log.info('Start computing distances and coordinates...')
        OASAdapterJSON.process_records(
            file_id=file_id,
            fn=fn_data_unit,
            fn_out=fn_data_unit_record,
            openfunc=prep_args['openfunc'],
            record_count=prep_args['record_count'],
            seq_id_field=seq_id_field,
            seq_field=seq_field,
            anchors=prep_args['anchors'],
            num_closest_anchors=num_closest_anchors,
            distance_measure_name=distance_measure,
            distance_measure_kwargs=distance_measure_kwargs,
            compression=output_file_compression,
            chunk_size=process_chunk_size,
            save_anchor_dists=save_anchor_dists,
            anchor_dist_compression=anchor_dist_compression,
            anchor_dist_compression_level=anchor_dist_compression_level,
            convert_json=seq_field_is_json,
            stop_after_n_chunks=0,
            nb_workers=process_nb_workers
        )
        log.info('Finished computing distances and coordinates.')

        # Write back the file status
        # TODO: Optimise, currently writes whole file.
        log.info(f'Updating file status to {FileStatus.completed}...')
        data_unit_item['status'] = FileStatus.completed
        with open(fn_file_list, 'w') as f:
            json.dump(files_to_process, f, indent=4, sort_keys=True)
        log.info(f'Finished updating file status.')

        # Show finished
        log.info(f'Finished processing {fn_data_unit}.')

    # Show finished
    log.info('-- Finished main loop. --')

    # Let user know some were skipped
    if skipped_files > 0:
        log.info(
            f'{skipped_files} file(s) were SKIPPED as previously processed. See log for details.')

    log.info('---- Finished Process OAS procedure. ----')


def log_file_processed(fn_log, sha1, fn_data_unit, status):
    """Keep track of which files have been processed"""
    with open(fn_log, 'at') as f:
        line = f'{sha1}, {status}, {fn_data_unit}'
        f.write(line + '\n')


def get_files_to_process(src_folder: str,
                         src_file_pattern: str) -> List[Dict[str, Any]]:
    """
    Get the sequence files to process and also generate a file ID for them.

    This file ID is later combined with a file-unique sequence ID,
    to create a sequence ID that is unique across all files in the
    environment. This may change between process runs and shouldn't be
    available to the user.

    Parameters
    ----------
    src_folder : str
        Folder containing the source sequence files.

    src_file_pattern : str
        Only include source files with this pattern.

    Returns
    -------
    List[Dict[str, Any]]
        List of Dictionaries:

        file_id: int
            Generated sys_file_id for the file (1-based). 
            Not persistent, do not reference outside of system.

        file_path: str
            Path to the file.

        status: FileStatus / str
            Processing state of the file.
    """

    # Loop through the list of files in the folder
    file_paths = glob.glob(
        os.path.join(src_folder, src_file_pattern)
    )

    # Sort
    file_paths = list(sorted(file_paths))

    # Build file list and IDs
    file_list = []
    file_id = 0
    for fn in file_paths:
        file_id += 1
        file_item = dict(
            file_id=file_id,
            file_path=fn,
            status=FileStatus.ready
        )
        file_list.append(file_item)

    # Return
    return file_list


# %% Main
if __name__ == '__main__':
    main(None)
    # main(['base/folder/ENV_NAME'])
