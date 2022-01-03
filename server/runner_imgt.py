# Run end-to-end processes

# %% Imports
import yaml
from pathlib import Path
import helpers
import airrmap.preprocessing.distance as distance
import logging
import os
import sys
import pandas as pd
import json
import airrmap.preprocessing.imgt as imgt
import argparse
from tqdm import tqdm
from typing import Any


# %%  Module to run all

def main(argv):
    parser = argparse.ArgumentParser(
        description='Build anchor db from imgt reference files.')
    parser.add_argument('envfolder', type=str,
                        help='Environment folder containing envconfig.yaml')

    args = parser.parse_args(argv)
    env_folder = args.envfolder

    # Load envconfig.yaml
    fn_config = os.path.join(env_folder, 'envconfig.yaml')
    with open(fn_config) as f:
        envcfg = yaml.load(f, Loader=yaml.FullLoader)
    distance_measure = envcfg['distance_measure']
    distance_measure_options = envcfg['distance_measure_options']
    cfg = envcfg['anchor']
    method = cfg['method']
    src_folder = cfg['src_folder']
    output_folder = cfg['build_folder']
    output_image_folder = cfg['build_image_folder']
    output_db_file = cfg['build_db_file']
    seq_field = cfg['seq_field']
    seq_field_is_json = cfg['seq_field_is_json']
    random_state = cfg['random_state']
    n_neighbors = cfg['n_neighbors'] if 'n_neighbors' in cfg else 15
 

    # Clear previous files
    helpers.clear_folder_or_create(os.path.join(env_folder,
                                                output_folder))
    helpers.clear_folder_or_create(
        os.path.join(env_folder, output_image_folder))

    # Process each file
    for fn_anchor_file in tqdm(cfg['src_files'], desc='Processing anchor files...'):

        
        process_imgt(env_folder=env_folder,
                     src_folder=src_folder,
                     imgt_ref_file=fn_anchor_file,
                     output_folder=output_folder,
                     output_image_folder=output_image_folder,
                     output_db_file=output_db_file,
                     seq_field=seq_field,
                     seq_field_is_json=seq_field_is_json,
                     dimension_reduction_method=method,
                     distance_measure=distance_measure,
                     distance_measure_kwargs=distance_measure_options,
                     random_state=random_state,
                     n_neighbors=n_neighbors)


def process_imgt(env_folder: str,
                src_folder: str, 
                 imgt_ref_file: str,
                 output_folder: str,
                 output_image_folder: str,
                 output_db_file: str,
                 seq_field: str,
                 seq_field_is_json: Any,
                 dimension_reduction_method: str,
                 distance_measure: str,
                 distance_measure_kwargs: dict,
                 random_state: int,
                 n_neighbors: int):



    # Get logger
    log = helpers.init_logger('applog',
                              os.path.join(env_folder,
                                           output_folder,
                                           'log.txt'))

    # Log start with passed in config
    log.info('Start Process IMGT procedure.')
    for k, v in locals().items():
        log.info(f'{k}: {v}')

    # Process the files (translate, filter)
    log.info('Start processing IMGT reference file.')
    df_imgt = imgt.parse_file(
        os.path.join(env_folder, src_folder, imgt_ref_file)
    )
    log.info('Finished processing IMGT reference files.')

    # Compute pairwise distance matrix
    log.info('Start computing pairwise distance matrix.')
    dist_function = getattr(distance, distance_measure)  # get by str name
    df_dist_matrix = imgt.build_distance_matrix(
        df_imgt, 
        distance_function=dist_function,
        measure_value=seq_field, 
        **distance_measure_kwargs
    )
    log.info('Finished computing pairwise distance matrices.')

    # Compute coordinates and distances
    log.info('Start computing coordinates.')
    df_coords, df_melted = imgt.compute_coords(
        df_dist_matrix, 
        method=dimension_reduction_method, 
        random_state=random_state,
        n_neighbors=n_neighbors)
    log.info('Finished computing coordinates.')

    # Generate plots
    log.info('Start generating plots.')
    log.info('Generating anchor distortion...')
    fig_anchor_distortion_group, fig_anchor_distortion_all = imgt.plot_anchor_distortion(
        df_melted)

    log.info('Generating anchor positions...')
    fig_anchor_positions = imgt.plot_anchor_positions(df_coords)
    log.info('Finished generating plots.')

    # Save the plots
    log.info('Saving plots...')
    fig_anchor_distortion_group.write_image(os.path.join(
        env_folder, output_image_folder, 'imgt_anchor_distortion_group.svg'))
    fig_anchor_distortion_all.write_image(os.path.join(
        env_folder, output_image_folder, 'imgt_anchor_distortion_all.svg'))
    fig_anchor_positions.write_image(os.path.join(
        env_folder, output_image_folder, 'imgt_anchor_positions.svg'))
    log.info('Finished saving plots.')

    # Save the data files
    # TODO: Change output name
    log.info('Saving data files...')
    # Convert dict to valid json if it was converted from json
    if seq_field_is_json != False:
        df_imgt[seq_field] = df_imgt[seq_field].apply(json.dumps)

    df_imgt.to_csv(os.path.join(env_folder, output_folder, 'IGHV_processed.csv'))
    df_melted.to_csv(os.path.join(env_folder, output_folder, 'IGHV_dist_melted.csv'))
    df_coords.to_csv(os.path.join(env_folder, output_folder, 'IGHV_coords.csv'))
    log.info('Finished saving data files.')

    # Re-read df_imgt and df_coords from file to get
    # dtypes compatible with SQLite
    # (dtypes are currently all object, which throws error with to_sql())
    # TODO: Find a better solution
    # convert_dtypes() changes all to string, with one object column for
    # aa_annotated
    df_imgt = pd.read_csv(os.path.join(env_folder, output_folder, 'IGHV_processed.csv'))
    df_coords = pd.read_csv(os.path.join(env_folder, output_folder, 'IGHV_coords.csv'))

    # Create db
    log.info(f'Writing anchors database...{output_db_file}')
    fn_db = os.path.join(env_folder, output_folder, output_db_file)

    log.info(f'Creating anchor_records table...')
    imgt.db_write_records(df_imgt, fn_db)
    log.info(f'Finished creating anchor_records table.')

    log.info(f'Creating anchor_coords table...')
    imgt.db_write_coords(df_coords, fn_db)
    log.info(f'Finished creating anchor_coords table.')

    log.info('Finished writing anchors database.')

    # Show finished
    log.info("Finished Process IMGT procedure.")


# %%
if __name__ == '__main__':
    main(None)
    #main(['base/folder/ENV_NAME'])

# %%
