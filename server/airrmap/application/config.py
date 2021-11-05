# Load config.yaml and convenience functions.

import yaml
import os
from enum import Enum
from typing import List, Dict, Any


class SeqFileType(Enum):
    META = 1
    RECORD = 2


class AppConfig():

    # TODO: Change default path
    def __init__(self, fn_config: str = r'/airrmap-data/config.yaml'):
        """
        Initialise and load configuration file.

        Parameters
        ----------
        fn_config : str, optional
            File path to config file, by default ''.
            If no path supplied, looks for
            config.yaml.
        """
        self.base_path: str = ''
        self.index_db: str = ''
        self.oas_list: str = '' # OAS Data Unit listing
        self.default_environment: str = ''
        self.tile_debug: bool = False
        self.loaded: bool = False

        self.facet_row_max_allowed: int = 10
        self.facet_col_max_allowed: int = 10

        self.column_plot_x: str = ''
        self.column_plot_y: str = ''
        self.column_value: str = ''
        self.column_class: str = ''
        self.default_num_bins: int = 256
        self.default_tile_size: int = 256
        self.default_statistic: str = ''

        self.coordinates_range: List[Any] = [0, 0]
        self.load(fn_config)

    def load(self, fn_config: str):
        """Load server settings from yaml"""
        with open(fn_config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.base_path = config['base_path']
            self.index_db = os.path.join(self.base_path, config['index_db'])
            self.oas_list = os.path.join(self.base_path, 'oas_listing.csv')
            self.default_environment = config['default_environment']
            self.tile_debug = config['tile_debug']

            self.facet_row_max_allowed = config['facet_row_max_allowed']
            self.facet_col_max_allowed = config['facet_col_max_allowed']

            self.column_plot_x = config['column_plot_x']
            self.column_plot_y = config['column_plot_y']
            self.column_value = config['column_value']
            self.column_class = config['column_class']
            self.default_num_bins = config['default_num_bins']
            self.default_tile_size = config['default_tile_size']
            self.default_statistic = config['default_statistic']

            self.coordinates_range = config['coordinates_range']
            self.loaded = True

        return self

    def get_anchordb(self, env_name: str) -> str:
        """Get anchor db path for env_name (case-sensitive)"""
        return os.path.join(
            self.base_path,
            env_name,
            'build',
            'anchor',
            'anchor.db'
        )

    def get_seqdb_folder(self, env_name) -> str:
        """Get the seqdb folder for an environment name"""
        return os.path.join(self.base_path, env_name, 'build', 'seq')

    def get_seqdb_parquet_folder(self, env_name) -> str:
        """Get the parquet folder for an environment"""
        return os.path.join(self.base_path, env_name, 'build', 'seq', 'records')

    def get_seqdbs(self, env_name: str) -> List[str]:
        """Get sequence dbs for env_name (case-sensitive)"""

        # Folder
        folder = os.path.join(
            self.base_path,
            env_name,
            'build',
            'seq',
        )

        # List of files ending .db
        fns = os.listdir(folder)
        fns = [
            os.path.join(folder, file)
            for file in fns
            if file.endswith('.db')
        ]

        # Return
        return fns

    def get_seq_folder(self, env_name: str, file_type: SeqFileType) -> str:
        """Get the parquet sequence record or meta folder for env_name (case-sensitive)"""

        # Folder
        folder = os.path.join(
            self.base_path,
            env_name,
            'build',
            'seq'
        )

        # Meta or Records folder
        if file_type == SeqFileType.META:
            folder = os.path.join(folder, 'meta')
        elif file_type == SeqFileType.RECORD:
            folder = os.path.join(folder, 'records')
        else:
            raise ValueError(f'Unknown file type: {str(file_type)}')

        return folder

    def get_seq_files(self, env_name: str, file_type: SeqFileType) -> List[str]:
        """Get the sequence record or meta files for env_name (case-sensitive)"""

        # Folder
        folder = self.get_seq_folder(env_name, file_type)

        # Meta or Records folder
        if file_type == SeqFileType.META:
            ext = '.meta.parquet'
        elif file_type == SeqFileType.RECORD:
            ext = '.parquet'
        else:
            raise ValueError(f'Unknown file type: {str(file_type)}')

        # List of files ending .parquet or .meta.parquet
        fns = os.listdir(folder)
        fns = [
            os.path.join(folder, file)
            for file in fns
            if file.endswith(ext)
        ]

        # Return
        return fns

    def get_env_folder(self, env_name) -> str:
        """Get the folder for an environment name"""
        return os.path.join(self.base_path, env_name)

    def get_env_config(self, env_name) -> Dict[str, Any]:
        """Load the envconfig.yaml file for the given environment"""

        fdr = self.get_env_folder(env_name)
        fn_config = os.path.join(fdr, 'envconfig.yaml')

        with open(fn_config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return config

    def get_envs(self) -> List[str]:
        """Get list of data environments"""

        # Ignore '.DS_Store' folder and
        # folders starting with '__'
        return [
            fdr
            for fdr in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, fdr))
            and not fdr.startswith('.')
            and not fdr.startswith('__')
        ]
