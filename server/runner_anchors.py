# Run end-to-end processes

# %% Imports
import yaml
import helpers
import airrmap.preprocessing.distance as distance
import os
import pandas as pd
import airrmap.preprocessing.anchors as anchors
import airrmap.preprocessing.distance as distance
import argparse
from tqdm import tqdm
from typing import Union

# %%  Module to run all

def main(argv):
    parser = argparse.ArgumentParser(
        description='Build anchor db from anchor sequences file.'
    )
    parser.add_argument(
        'envfolder',
        type=str,
        help='Environment folder containing envconfig.yaml'
    )

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
    seq_id_field = cfg['seq_id_field']
    group_field = cfg['group_field']
    random_state = cfg['random_state']
    n_neighbors = cfg['n_neighbors'] if 'n_neighbors' in cfg else 15

    # Clear previous files
    helpers.clear_folder_or_create(os.path.join(env_folder,
                                                output_folder))
    helpers.clear_folder_or_create(
        os.path.join(env_folder, output_image_folder))

    # Process each file
    for fn_anchor_file in tqdm(cfg['src_files'], desc='Processing anchor files...'):

        process_anchors(env_folder=env_folder,
                        src_folder=src_folder,
                        anchor_file=fn_anchor_file,
                        output_folder=output_folder,
                        output_image_folder=output_image_folder,
                        output_db_file=output_db_file,
                        seq_field=seq_field,
                        seq_id_field=seq_id_field,
                        group_field=group_field,
                        dimension_reduction_method=method,
                        distance_measure=distance_measure,
                        distance_measure_kwargs=distance_measure_options,
                        random_state=random_state,
                        n_neighbors=n_neighbors)


def process_anchors(env_folder: str,
                    src_folder: str,
                    anchor_file: str,
                    output_folder: str,
                    output_image_folder: str,
                    output_db_file: str,
                    seq_field: str,
                    seq_id_field: str,
                    group_field: Union[str, None],
                    dimension_reduction_method: str,
                    distance_measure: str,
                    distance_measure_kwargs: dict,
                    random_state: int,
                    n_neighbors: int):

    # Get logger
    log = helpers.init_logger('applog',
                              os.path.join(env_folder,
                                           output_folder,
                                           'log.txt')
                              )

    # Log start with passed in config
    log.info('Start Process Anchors procedure.')
    for k, v in locals().items():
        log.info(f'{k}: {v}')

    # Load anchor file
    log.info('Start loading anchor file.')
    df_anchors = pd.read_csv(
        os.path.join(
            env_folder,
            src_folder,
            anchor_file
        ),
        index_col=seq_id_field
    )
    log.info('Finished loading anchor file.')

    # Compute pairwise distance matrix
    log.info('Start computing pairwise distance matrix.')
    dist_function = getattr(distance, distance_measure)  # get by str name
    df_dist_matrix = distance.create_distance_matrix(
        records=df_anchors,
        distance_function=dist_function,
        measure_value=seq_field,
        **distance_measure_kwargs
    )
    log.info('Finished computing pairwise distance matrices.')

    # Triangle inequality test (check distance function is valid)
    log.info('Start triangle inequality test.')
    triangle_test = distance.verify_triangle_inequality(
        df_dist_matrix,
        n=10000)
    if len(triangle_test) > 0:
        err_msg = 'Triangle inequality does not hold! Check distance measure.'
        log.critical(err_msg)
        print(triangle_test)
        raise Exception(err_msg)
    log.info('Finished triangle inequality test.')

    # Compute coordinates and distances
    log.info('Start computing coordinates.')
    df_coords, df_melted = anchors.compute_coords(
        df_dist_matrix,
        method=dimension_reduction_method,
        random_state=random_state,
        n_neighbors=n_neighbors)
    log.info('Finished computing coordinates.')

    # Add group field to DataFrames if provided
    seq_id_group_map = None
    if group_field is not None:
        seq_id_group_map = {k:v for k, v in zip(df_anchors.index, df_anchors[group_field])}

    # Generate plots
    log.info('Start generating plots.')
    log.info('Generating anchor distortion...')
    fig_anchor_distortion_group, fig_anchor_distortion_all = anchors.plot_anchor_distortion(
        df_melted=df_melted,
        seq_id_group_map=seq_id_group_map
    )

    log.info('Generating anchor positions...')
    fig_anchor_positions = anchors.plot_anchor_positions(
        df_coords=df_coords,
        seq_id_group_map=seq_id_group_map)
    log.info('Finished generating plots.')

    # Save the plots
    log.info('Saving plots...')
    fig_anchor_distortion_group.write_image(os.path.join(
        env_folder, output_image_folder, 'anchor_distortion_group.svg'))
    fig_anchor_distortion_all.write_image(os.path.join(
        env_folder, output_image_folder, 'anchor_distortion_all.svg'))
    fig_anchor_positions.write_image(os.path.join(
        env_folder, output_image_folder, 'anchor_positions.svg'))
    log.info('Finished saving plots.')

    # Save the data files
    log.info('Saving data files...')
    df_anchors.to_csv(os.path.join(
        env_folder, output_folder, 'anchor_processed.csv'))
    df_melted.to_csv(os.path.join(
        env_folder, output_folder, 'anchor_dist_melted.csv'))
    df_coords.to_csv(os.path.join(
        env_folder, output_folder, 'anchor_coords.csv'))
    log.info('Finished saving data files.')

    # Re-read df_coords from file to get
    # dtypes compatible with SQLite
    # (dtypes are currently all object, which throws error with to_sql())
    # TODO: Find a better solution
    # convert_dtypes() changes all to string, with one object column for
    # aa_annotated
    df_coords = pd.read_csv(
        os.path.join(
            env_folder,
            output_folder,
            'anchor_coords.csv'
        )
    )

    # Create db
    log.info(f'Writing anchors database...{output_db_file}')
    fn_db = os.path.join(env_folder, output_folder, output_db_file)

    log.info(f'Creating anchor_records table...')
    anchors.db_write_records(df_anchors, fn_db)
    log.info(f'Finished creating anchor_records table.')

    log.info(f'Creating anchor_coords table...')
    anchors.db_write_coords(df_coords, fn_db)
    log.info(f'Finished creating anchor_coords table.')

    log.info('Finished writing anchors database.')

    # Show finished
    log.info("Finished Process Anchors procedure.")


# %%
if __name__ == '__main__':
    main(None)
    #main (['/airrmap-data/ENV_NAME'])