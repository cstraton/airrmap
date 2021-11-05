"""Functionality for interacting with OAS."""

# Imports
import urllib
import json
import re
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


class OASHelper:
    """
    Functionality for interacting with OAS.

    See also runner_oas_listing.py

    Typical use:
    1. Simulate a POST search request with all search fields left blank.
    2. Parse the full results table and collect Bulk Download file links.
    3. Open the Details page and collect the download link and metadata.
    4. Perform a CrossRef lookup using doi to retrieve additional metadata
        such as the study title.

    Sense check:
    Visit http://opig.stats.ox.ac.uk/webapps/oas/oas, and run a search
    with all form fields left blank/*. The total number of entries shown
    at the bottom should match the number of urls found by this script.

    Notes:
    1. Only unpaired sequences are currently supported.
    2. If changing, update runner_oas_listing.py
    """

    # OAS Url, include final '/'
    OAS_BASE_URL = 'http://opig.stats.ox.ac.uk/webapps/oas/'
    NO_LINK = 'no link'

    def __init__(self):
        self._crossref_cache: Dict = {}

    @staticmethod
    def _cxref_prep(message: Any,
                    key: Any,
                    el_index: Optional[int] = None,
                    remove_tags: bool = False) -> Any:
        """
        Preprocess message values from the CrossRef service.

        Parameters
        ----------
        message : Any
            The CrossRef message result for a given DOI.

        key : Any
            The property in the message to process.

        el_index : Optional[int], optional
            If the property value is a sequence, then the
            element index to return. By default, None.

        remove_tags : bool, optional
            Remove XML tags from the value, by default False.

        Returns
        -------
        Any
            The processed value, or None if there was no value.
        """

        # Check property exists
        if not key in message:
            return None

        # If trying to get an element of the array, then
        # check it isn't empty (properties with no value
        # for this record may have [] instead of [['val1', 'val2'...]])
        if el_index is not None:
            if len(message[key]) > 0:
                result = message[key][el_index]
            else:
                result = None
        else:
            result = message[key]

        # Remove tags if required (replace with space)
        if remove_tags == True:
            sp = BeautifulSoup(result, 'lxml')
            result = sp.get_text(separator=' ')

        # Return
        return result

    @staticmethod
    def load_search_results() -> str:
        """
        POST a blank form request to retrieve a list of all data units.
        """

        # Construct request
        # (See Network)
        url = OASHelper.OAS_BASE_URL + 'oas'  # unpaired seqs url
        post_data = {
            'Chain': '*',
            'Isotype': '*',
            'Age': '*',
            'Disease': '*',
            'BSource': '*',
            'BType': '*',
            'Longitudinal': '*',
            'Species': '*',
            'Vaccine': '*',
            'Subject': '*'
        }

        # Load result
        result = requests.post(url, data=post_data)

        # Return
        return result.text

    @staticmethod
    def parse_results_table(search_results: str) -> pd.DataFrame:
        """
        Parse the results table from the search results.

        The table includes all units in OAS.
        This parses the HTML and the main results table, 
        one row for each Data Unit. 

        Parameters
        ----------
        search_results : str
            The search results HTML content.

        Returns
        -------
        pd.DataFrame
            The results table. An additional 'Details link'
            column is added for the original 'Details' url.
        """

        # Example page (click Search to see results):
        # http://opig.stats.ox.ac.uk/webapps/oas/oas

        # Adapted from useRj (2017)
        # "How to preserve links when scraping a table with beautiful soup and pandas"
        # https://stackoverflow.com/a/42294689
        sp = BeautifulSoup(search_results, 'lxml')
        tb = sp.find('table', attrs={'id': 'results_table'})
        df = pd.read_html(str(tb), encoding='utf-8', header=0)[0]
        df['Details link'] = [
            OASHelper.OAS_BASE_URL + tag.get('href')
            for tag in tb.find_all('a')
        ]

        # Return
        return df

    @staticmethod
    def parse_details_page(details_page: str) -> Dict[str, Any]:
        """
        Parse a unit's Details page.

        Parameters
        ----------
        details_page : str
            The Details page HTML content.

        Returns
        -------
        Dict
            Data unit details including the download link.
        """

        # Example page:
        # http://opig.stats.ox.ac.uk/webapps/oas/dataunit?unit=Eliyahu_2018/csv/ERR2843400_Heavy_IGHE.csv.gz

        # Parse
        sp = BeautifulSoup(details_page, 'lxml')

        # Get the Data Unit url
        data_unit_download = sp.find(text='Download the data-unit: ')
        data_unit_link = data_unit_download.findNext('a').get('href')

        # Parse the table (Property, Value)
        # Adapted from useRj (2017)
        # "How to preserve links when scraping a table with beautiful soup and pandas"
        # https://stackoverflow.com/a/42294689
        tb = sp.find('table', attrs={'id': 'results_table'})
        df = pd.read_html(str(tb), encoding='utf-8', header=0)[0]

        # Create the record
        details = {k: v for k, v in zip(df['Parameter'], df['Value'])}
        details['Download link'] = data_unit_link

        # Overwrite 'here' text with link url
        # (Should just be one link)
        details['Link to study'] = [
            tag.get('href')
            for tag in tb.find_all('a') if tag.string == 'here'
        ][0]

        # Return
        return details

    def get_file_links(self, search_results: str) -> List[str]:
        """
        Extract the file urls from the search results.

        This links are used by the Bulk Download option and
        are separate to the results table. They may also be 
        used as a checksum to ensure all units have been
        captured from the results table.

        Parameters
        ----------
        search_results : str
            The raw HTML content from the search results page.

        Returns
        -------
        List[str]
            The list of file links.
        """

        # The file links are between the start_line and end_line values.
        # Example:
        # var CSV = [
        # "wget http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Eliyahu_2018/csv/ERR2843418_Heavy_IGHA.csv.gz",
        # ...
        # ].join('\n')
        start_line = 'var CSV = ['
        end_line = "].join('\\n')"
        line_contains = 'wget http'
        line_replace = 'wget '
        read_enabled = False
        file_urls = []

        # Loop through lines
        # If start line found, start reading from next line.
        # If end line found, stop.
        for line in search_results.splitlines():
            line_clean: str = line.strip()
            if start_line in line_clean:
                read_enabled = True
                continue
            elif end_line in line_clean:
                read_enabled = False
                break

            if read_enabled and line_contains in line_clean:
                # Replacements
                line_clean = line_clean.replace(line_replace, '', 1)
                line_clean = line_clean.replace('"', '')
                line_clean = line_clean.replace(',', '')

                # Add to results
                file_urls.append(line_clean)

        # Return
        return file_urls

    @staticmethod
    def read_metadata(url: str) -> Dict[Any, Any]:
        """
        Read the metadata line of a single OAS Data Unit file.

        Adapted from:
        http://opig.stats.ox.ac.uk/webapps/oas/documentation

        Parameters
        ----------
        url : str
            The OAS url of the Data Unit file.

        Returns
        -------
        Dict
            The metadata.
        """

        metadata = ','.join(pd.read_csv(url, nrows=0).columns)
        return json.loads(metadata)

    @staticmethod
    def extract_doi(url: str) -> Any:
        """
        Extract the doi value from a doi link.

        Parameters
        ----------
        url : str
            DOI link, e.g. https://doi.org/10.3389/fimmu.2018.03004

        Returns
        -------
        None
            If a non-doi link was passed.
        str
            The doi value, e.g. '10.3389/fimmu.2018.03004'.
        """

        # Look for anything after https://doi.org/
        doi_match = re.match(r'^.*doi.org\/(.*)$', url)
        if doi_match:
            return doi_match.group(1).strip()
        else:
            return None

    @staticmethod
    def extract_study_name(url: str) -> Any:
        """
        Extract the study name from the Details link.

        Parameters
        ----------
        url : str
            The url to a unit's 'Details' page.

        Returns
        -------
        str, None
            The study name if found in the url, otherwise None.
        """

        # Example (extract 'Eliyahu_2018', look for 'unit=<study name>/'):
        # http://opig.stats.ox.ac.uk/webapps/oas/dataunit?unit=Eliyahu_2018/csv/ERR2843400_Heavy_IGHE.csv.gz
        study_name_match = re.match(r'^.*unit=(.*?)\/.*$', url)
        if study_name_match:
            return study_name_match.group(1).strip()
        else:
            return None

    def get_crossref(self, doi: str) -> Dict[str, Any]:
        """
        Get article metadata using DOI and CrossRef API.

        Caches results to prevent duplicate calls.

        Parameters
        ----------
        doi : str
            DOI reference.

        Returns
        -------
        Dict[str, Any]
            Dictionary with CrossRef metadata.
        """

        # Return cached version if exists
        if doi in self._crossref_cache:
            return self._crossref_cache[doi]

        # Make CrossRef call
        url = f'https://api.crossref.org/works/{doi.strip()}/'
        with urllib.request.urlopen(url) as f:
            response = f.read()
        data = json.loads(response)
        message = data['message']

        # List all keys (debugging)
        # for k, v in message.items():
        #    print(f'{k}\n{v}\n------')

        # Get name of first author
        first_author = message['author'][0]
        if 'name' in first_author:
            first_author = f"{first_author['name']}"  # Organisation
        elif 'given' in first_author and 'family' in first_author:
            first_author = f"{first_author['given']} {first_author['family']}"
        else:
            first_author = f'ERROR: Could not get first author: {str(first_author)}'

        # Get name of last author
        last_author = message['author'][-1]
        if 'name' in last_author:
            last_author = f"{last_author['name']}"  # Organisation
        elif 'given' in last_author and 'family' in last_author:
            last_author = f"{last_author['given']} {last_author['family']}"
        else:
            last_author = f'ERROR: Could not get last author: {str(last_author)}'

        # Build result
        result = dict(
            cxref_DOI=self._cxref_prep(
                message,
                'DOI'
            ),
            cxref_DOILink=self._cxref_prep(
                message,
                'URL'
            ),
            cxref_FirstAuthor=first_author,
            cxref_LastAuthor=last_author,
            cxref_Title=self._cxref_prep(
                message,
                'title',
                el_index=0
            ),
            cxref_Publisher=self._cxref_prep(
                message,
                'publisher'
            ),
            cxref_ContainerTitle=self._cxref_prep(
                message,
                'container-title',
                el_index=0
            ),
            cxref_IsReferencedByCount=self._cxref_prep(
                message,
                'is-referenced-by-count'
            ),
            cxref_Abstract=self._cxref_prep(
                message,
                'abstract',
                remove_tags=True
            )
        )

        # Cache
        self._crossref_cache[doi] = result

        # Return
        return result

    def build_meta_record(self, details_url: str) -> Dict:
        """
        Build a complete metadata record for a single OAS Data Unit.

        1. Load the details from the 'Details' page.
        2. Add DOI (extract from study link).
        3. Call CrossRef service to get title and other details
            (uses caching to avoid duplicate calls for the same DOI).

        Parameters
        ----------
        details_url : str
            The url for the Data Unit's 'Details' page.

        Returns
        -------
        Dict
            Metadata record for the given Data Unit.
        """

        # Collect information from Details page
        details_html = requests.get(details_url).text
        details_record = self.parse_details_page(details_html)

        # Add Details page link
        details_record['Details link'] = details_url

        # Get study name from the link (e.g. 'Eliyahu_2018')
        details_record['Study name'] = self.extract_study_name(details_url)

        # Add DOI to the record
        doi = self.extract_doi(details_record['Link to study'])
        details_record['DOI'] = doi

        # Enrich with CrossRef information
        if doi is not None:
            crossref_meta = self.get_crossref(doi)
            details_record.update(crossref_meta)

        # Return
        return details_record
