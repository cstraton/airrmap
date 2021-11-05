"""Re-build the OAS list, collect metadata for all Data Units."""

# Imports
import sys
import argparse
import pandas as pd
import os
import helpers
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from joblib import Parallel, delayed
from airrmap.application.config import AppConfig, SeqFileType
from airrmap.util.oas_helper import OASHelper
from typing import List


def get_metadata_record(
        i: int,
        detail_urls: List[str],
        oas_helper: OASHelper,
        log: logging.Logger):
    """
    Collect metadata record for given unit.

    Separate function to support Parallel running.
    """

    # Get meta data
    url = detail_urls[i]
    log.info(f"Collecting metadata for '{url}'.")
    metadata = oas_helper.build_meta_record(url)
    log.info(f"Finished collecting metadata for '{url}'.")
    return metadata


def main(argv):

    # Init
    appcfg = AppConfig()
    fn_oas_listing = appcfg.oas_list
    oas_helper = OASHelper()
    nthreads_default = 6

    # Process args
    parser = argparse.ArgumentParser(
        description='Refresh the OAS list of Data Units and collect associated metadata.'
    )
    parser.add_argument(
        '--nrows',
        type=int,
        help='Number of rows from the search results to process, before stopping. Primarily for testing.'
    )
    parser.add_argument(
        '--nthreads',
        type=int,
        help=f'Number of threads to use, default is {nthreads_default}.'
    )
    parser.add_argument(
        '--clearlog',
        action='store_true',
        help='Clear the previous log file instead of appending to it.'
    )

    args = parser.parse_args(argv)
    nrows = args.nrows if args.nrows else None
    nthreads = args.nthreads if args.nthreads else nthreads_default
    clear_log = args.clearlog
    fn_log = os.path.join(appcfg.base_path, 'oas_listing.log')
    log_cleared = False

    # Clear previous log if it exists
    if clear_log and os.path.exists(fn_log):
        os.remove(fn_log)
        log_cleared = True

    # Get logger
    log = helpers.init_logger(
        'oaslistlog',
        fn_log
    )

    # Log start with passed in config
    log.info('--------------------------------------')
    if log_cleared:
        log.info('Previous log file was cleared.')

    log.info('Start Process OAS listing.')
    for k, v in locals().items():
        log.info(f'{k}: {v}')

    # Perform OAS search
    log.info(f'Performing OAS search...')
    log.info(f"OAS base path is '{oas_helper.OAS_BASE_URL}'")
    search_results = oas_helper.load_search_results()
    log.info(f'Finished OAS search.')

    # Get Data Unit links (using in-page list for bulk download link)
    log.info(f'Getting Data Unit file links...')
    file_links = oas_helper.get_file_links(search_results)
    log.info(
        f'Finished getting Data Unit file links, {len(file_links)} found.')

    # Check links found
    if len(file_links) <= 0:
        log.critical('No file links were found, stopping.')
        sys.exit(1)

    # Parse results table
    log.info(f'Parsing results table...')
    df_search_results = oas_helper.parse_results_table(search_results)
    log.info(
        f'Finished parsing results table, {len(df_search_results)} record(s).')

    # Check number of records in the results table matches
    # the list of in-page download links for the bulk download
    if len(df_search_results) != len(file_links):
        log.critical(
            f'Results table ({len(df_search_results)}) vs ' +
            f'Bulk Download list ({len(file_links)}) mismatch.'
        )
        sys.exit(1)
    else:
        log.info(f'Results table vs Bulk Download list checksum OK.')

    # Get links to each unit's 'Details' page
    detail_urls = list(df_search_results['Details link'])

    # Build meta records
    if nrows is None:
        nrows = len(detail_urls)

    log.info(f'Number of rows that will be processed: {nrows}')
    log.info(f'Number of threads to use: {nthreads}')
    i = 0

    # Use threading instead of multiprocessing
    # so that the OASHelper instance and CrossRef cache is shared.
    # REF: https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
    with logging_redirect_tqdm():
        meta_records = Parallel(n_jobs=nthreads, backend='threading')(
            delayed(get_metadata_record)(i, detail_urls, oas_helper, log) for i in tqdm(range(nrows))
        )

    # Place in Pandas DataFrame and move Study name to front
    df = pd.DataFrame.from_records(meta_records)
    study_name = df.pop('Study name')
    df.insert(0, study_name.name, study_name)
    log.info(f'Collected {len(meta_records)} metadata record(s).')

    # Save
    log.info(f"Saving to '{fn_oas_listing}'...")
    df.to_csv(fn_oas_listing, index=False)
    sha1 = helpers.sha1file(fn_oas_listing)
    log.info(f"SHA-1 is '{sha1}'")
    log.info(f"Finished saving to '{fn_oas_listing}'.")

    # Final check
    if len(df) != nrows:
        log.critical(
            f'Some records missing! Expected {nrows}, but only have {len(df)}. Try re-running.')
        sys.exit(1)
    else:
        log.info(f'Final checksum OK: {nrows} record(s) saved.')

    # Show finished
    log.info("Finished Process OAS listing.")


if __name__ == '__main__':
    main(None)
