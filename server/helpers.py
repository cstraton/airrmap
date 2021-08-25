"""
General helper functions.
"""

# %% imports
import pandas as pd
import cProfile
import logging
import hashlib
import shutil
import pstats
import gzip
import yaml
import sys
import re
import os
from io import StringIO


# %%
def read_json_nohdr(fn: str, header_rows: int = 1) -> pd.DataFrame:
    """Read json file minus the header"""

    with open(fn, 'r') as f:
        lines = f.readlines()

    #header = lines[0]
    records = lines[header_rows:]
    return pd.read_json('\n'.join(records), lines=True)


# %%
def clear_folder_or_create(path: str) -> str:
    """
    Clear existing files or create the folder.

    Only deletes immediate files (not subdirs) to
    prevent accidental deletion of a large set of files.
    """

    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=False)

    else:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  #  delete or unlink file

    return path


def init_logger(name: str, log_file_path: str) -> logging.Logger:
    """Set up a logger with the given name.

    The logger can be accessed from any module with the given name
    (no need to pass around the logger instance).

    Args:
        name (str): Unique name for the logger instance.
        log_file_path (str): File path to store log output.

    Example:
        ```
        init_logger('my logger', 'logs/my_log.log')

        import logging
        logger = logging.getLogger('my logger')
        ```

    Returns:
        logging.Logger: The logger instance
    """

    # Adapted from:
    # REF: https://timber.io/blog/the-pythonic-guide-to-logging/

    # Set up logger
    logFormatter = '%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s'
    logging.basicConfig(
        format=logFormatter,
        level=logging.DEBUG
    )
    logger = logging.getLogger(name)

    # Write to handler
    handler = logging.FileHandler(log_file_path)
    handler.setFormatter(logging.Formatter(logFormatter))
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger


# %%
def sha1file(fn: str) -> str:
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


def load_defaults() -> dict:
    """Load default/test settings"""
    fn = os.path.join(os.path.dirname(__file__), 'defaults.yaml')
    with open(fn) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def performance_profile(func, output_fn: str, repeat=1, **kwargs):
    """
    Run function and measure performance.
    """

    # Adapted from:
    # https://stackoverflow.com/questions/51536411/saving-cprofile-results-to-readable-external-file

    pr = cProfile.Profile()
    pr.enable()
    for i in range(repeat):
        func(**kwargs)
    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats()
    with open(output_fn, 'w+') as f:
        f.write(s.getvalue())

    return output_fn
