# Get listing of OAS studies
# Requires oas_doi_lookup.csv (maps article links in OAS to doi numbers,
# so metadata can be obtained from the CrossRef service

# Currently OAS enables searching by study parameters, but does
# not provide a list of the studies (only links to them)

# Process
# 1. Gather list of folders for each study
# 2. List data unit files in each one
# 3. Open the first file (gzip)
# 4. Get the first line (always json, even if file is csv or json)
# 5. Get the link property
# 6. Use the oas_doi_lookup.csv file to map links to doi (manually produced)
# 7. Use CrossRef service to get title and author


# Main source (unpaired sequences)
# http://opig.stats.ox.ac.uk/webapps/ngsdb/json/


# Meta for each dataset
#
# http://opig.stats.ox.ac.uk/webapps/ngsdb/meta/

# Last modified:
# 2021-01-14 (see Buchheim_2020)

# JSON metadata all
# http://opig.stats.ox.ac.uk/webapps/ngsdb/json/oas_metadata.json

#  "Currently three DOI registration agencies have implemented content negotation for their DOIs: CrossRef, DataCite and mEDRA."
#  https://stackoverflow.com/questions/10507049/get-metadata-from-doi

# %%
import urllib
import re
import requests
from bs4 import BeautifulSoup
import gzip
import pandas as pd
import json
from tqdm import tqdm

# %% Read oas_metadata.json (out of data)
# fn = r'http://opig.stats.ox.ac.uk/webapps/ngsdb/json/oas_metadata.json'
# df = pd.read_json(fn, orient='index')
# df.to_csv('oas_listing.csv')


# %%
def get_crossref(doi: str):
    """Get article metadata using doi and CrossRef API

    Args:
        doi (str): doi reference

    Returns:
        dict: Dictionary with title, first and last authors
    """

    url = f'https://api.crossref.org/works/{doi.strip()}/'

    with urllib.request.urlopen(url) as f:
        response = f.read()

    data = json.loads(response)
    message = data['message']

    # for k, v in message.items():
    #    print(k)

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

    result = dict(doi=doi,
                  first_author=first_author,
                  last_author=last_author,
                  title=message['title'][0],
                  doi_url=message['URL'],
                  publisher=message['publisher'],
                  is_referenced_by_count=message['is-referenced-by-count'])

    return result


# %%
def get_hyperlinks(url: str, p: re.Pattern) -> list[str]:
    """Get list of hyperlinks from web page

    Args:
        url (str): URL to web page
        p (re.Pattern): RegEx pattern to match

    Returns:
        list[str]: List of hyperlinks
    """

    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    filelist = [url + node.get('href')
                for node in soup.find_all('a') if p.match(node.get('href'))]

    return filelist

# %%


def read_meta_file(url) -> str:
    """Reads the json metadata file

    Args:
        url (str): URL to json metadata file

    Returns:
        str: The json string
    """

    with urllib.request.urlopen(url) as f:
        s = f.read()

    return s


def get_first_line(url) -> str:
    """Get first line from gzipped file

    Opens the gzipped url and reads the first line
    (doesn't need to download the whole file.)

    Args:
        url (str): URL of gzipped file.

    Returns:
        str: First line from the file.
    """

    with urllib.request.urlopen(url) as f1:
        with gzip.open(f1) as f2:
            s = f2.readline()

    return s.decode('UTF-8')


# %% Init
url_base = r'http://opig.stats.ox.ac.uk/webapps/ngsdb/meta/'
if not url_base.endswith('/'):
    url_base = url_base + '/'


# %% Get list of directories
# Look for _YYYY in hyperlink
pdirs = re.compile(r'.*_\d{4}')
dirlist = get_hyperlinks(url_base, pdirs)


# %% Go through folders and meta files
study_list = {}
data_unit_list = []
data_unit_meta = []
pfiles = re.compile(r'.*\.json$')
hrefs = None
for dirurl in tqdm(dirlist, desc="Reading study folders..."):

    # (e.g. '.../Wesemann_2013/' -> 'Wesemann_2013')
    study_name = dirurl.split('/')[-2]
    study_year = study_name.split('_')[-1][:4]
    file_list = get_hyperlinks(dirurl, pfiles)
    data_unit_list.extend(file_list)

    # Go through each json meta file
    for fileurl in tqdm(file_list, desc="Reading meta json files..."):
        meta_s = read_meta_file(fileurl)
        meta_d = json.loads(meta_s)
        meta_d['oas_metafolder'] = dirurl
        meta_d['oas_metafile'] = fileurl  # add on source of information
        meta_d['study_name'] = study_name
        meta_d['year'] = study_year
        data_unit_meta.append(meta_d)

    # List of studies with selected metadata
    # Assume these fields are the same for each study
    # so just take from the last file.
    study_list[dirurl] = dict(study_name=study_name, study_year=study_year, author=meta_d['Author'],
                              oas_metafolder=dirurl, article_link=meta_d['Link'], species=meta_d['Species'],
                              vaccine=meta_d['Vaccine'], disease=meta_d['Disease'])


# %% Save Data Unit metadata
df_data_unit_meta = pd.DataFrame.from_records(data_unit_meta)
df_data_unit_meta.index.name = 'index'
df_data_unit_meta.to_csv('oas_data_units.csv')

# %% Save the study list
df_study_list = pd.DataFrame.from_dict(study_list, orient='index')
df_study_list.index.name = 'index'
df_study_list.to_csv('oas_study_list.csv')


# %% Retrieve additional metadata using doi lookup
# Requires that every link in oas has a doi mapping
# in oas_doi_lookup.csv
# Example:
#   doi = '10.1016/j.jaci.2015.09.027'
#   get_crossref(doi)

# %%
df_doi = pd.read_csv('oas_doi_lookup.csv')
doi_records = []
doi = ''
for i in tqdm(range(len(df_doi.index)), desc="Getting doi metadata..."):
    row = df_doi.iloc[i]
    doi = row['doi']
    meta = get_crossref(doi)
    meta['article_link'] = row['url']
    doi_records.append(meta)


# %% Save
df_doi_results = pd.DataFrame.from_records(doi_records)
df_doi_results.to_csv('oas_doi_results.csv')


# %%
