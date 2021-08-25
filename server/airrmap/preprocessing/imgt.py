# Functionality for processing of IMGT VDJ reference sequences
# (anchor sequences)

# tests > test_imgt.py

# %% Imports
import pandas as pd
import os
import sqlite3
import json
# import umap (performed further down)
import plotly.express as px
from plotly.graph_objs._figure import Figure
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
from sklearn import manifold
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from typing import Dict, Tuple, Sequence, Union

import airrmap.preprocessing.distance as distance

# ----------- PARSING -----------


def parse_file(fn: str) -> pd.DataFrame:
    """Parse anchors fasta or json file"""
    if fn.lower().endswith('.fasta'):
        return parse_fasta_file(fn)
    elif fn.lower().endswith('.json'):
        return parse_json_file(fn)
    else:
        raise Exception(
            f'Expecting .fasta or .json file: "{fn}"".'
        )


def parse_fasta_file(fn: str) -> pd.DataFrame:
    """Parse reference file of nucleotide sequences

    Args:
        fn (str): File of IMGT V/D/J alleles (DNA)

    Returns:
        DataFrame: DataFrame of parsed and filtered records, 
            index = allele name (e.g. IGHV-10-1*02).
            aa = ungapped amino acid sequence.
            aa_gapped = gapped amino acid sequence.
            If aa translation error, aa_err will contain error message.
    """

    records = {}
    for record in SeqIO.parse(fn, 'fasta'):
        record_parsed = _parse_sequence_record(record)
        records[record_parsed['name']] = record_parsed

    # TODO: Optimize and log to file
    # -------------------------------

    # Filter out aa translation errors
    records = {k: v for k, v in records.items() if not 'aa_err' in v}

    # Filter out sequences with "*"
    records = {k: v for k, v in records.items() if not '*' in v['aa_gapped']}

    # Filter out partial records ("..........")
    records = {k: v for k, v in records.items(
    ) if not '..........' in v['aa_gapped']}

    #  %% Double check for errors
    for k, record in records.items():
        if 'aa_err' in record:
            raise Exception('Found untranslated IMGT reference: ', k)

    # Load to DataFrame
    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = 'index'

    # Add on annotated sequences - remove gaps '.' from annotated seq
    # as OAS doesn't include gaps in numbered sequence.
    # TODO: Currently only applicable for V heavy germline gene
    df['aa_annotated'] = df['aa_gapped'].apply(
        _parse_v_germline_to_imgt, chain='h', remove_gaps=True)

    return df


def parse_json_file(fn: str) -> pd.DataFrame:
    """
    Parse json file with annotated/numbered sequence.

    Parameters
    ----------
    fn : str
        A json records file with minimum 'name' and [seq_field]
        fields (see example). File may contain additional fields.


    Returns
    -------
    pd.DataFrame
        The populated DataFrame containing the sequences.

    Example
    -------
    ```
    {"name": 1, "aa_annotated": {"fwh1": {"1": "M", "2", "L"...}}}
    {"name": 2, "aa_annotated": {"fwh1": {"1": "M", "2", "L"...}}}
    ```
    """

    df: pd.DataFrame = pd.read_json(fn, lines=True)

    if (not 'name' in df.columns):
        raise ValueError(
            f'Records must have "name" field.'
        )

    # Set index to be the name

    df['index'] = df['name']
    df.set_index('index', inplace=True)
    df.index.name = 'index'

    return df


def _parse_sequence_record(record):
    """Parse header fields and translate"""

    record_parsed = _parse_description(record.description)
    record_parsed['dna'] = str(record.seq)

    # Only translate from codon_start
    codon_start = 0
    if record_parsed['codon_start'].isnumeric():
        codon_start = int(record_parsed['codon_start']) - 1  # 1 based

    try:
        record_parsed['aa_gapped'] = str(
            record[codon_start:].translate(gap='.').seq)
        record_parsed['aa'] = record_parsed['aa_gapped'].replace('.', '')
    except Exception as err:
        record_parsed['aa_err'] = err
    return record_parsed


def _parse_description(description):
    """Split FASTA description value to fields"""

    # Fields: http://www.imgt.org/genedb/GENElect?query=7.3+IGHV&species=Homo+sapiens
    field_names = (
        'accession_number',  # 1
        'name',             # 2
        'species',          # 3
        'functionality',    # 4
        'region',           # 5
        'start_end_pos',    # 6
        'nucleotide_count',  # 7
        'codon_start',      # 8
        'nt_add_5prime',    # 9
        'nt_add_remove_3prime',  # 10
        'nt_add_remove_seq_correction',  # 11
        'aa_count',         # 12
        'char_count',       # 13
        'partial',          # 14
        'reverse_complementary',  # 15
        'unused1'         # 16
    )

    # Split values - ensure expected number of fields
    values = description.split('|')
    if len(values) != len(field_names):
        raise Exception('unexpected number of values in description ' +
                        '(%i values, %i fields)' % (len(values), len(field_names)))

    # Return
    return dict(zip(field_names, values))


def get_region_positions(chain: str, as_dict=False, is_rearranged=False, for_gapped_seq=False):
    """
    Get IMGT region positions (as annotated in OAS).

    Parameters
    ----------
    chain : str
        The chain ID, e.g. 'h' for heavy. As annotated for the numbered
        sequence in OAS.

    as_dict : bool, optional
        True to return as a dictionary of tuples, or False as a tuple of tuples. 
        By default False.

    is_rearranged : bool, optional
        Affects CDR3 numbering and inclusion of fw4. 
        If germline sequence, pass False (CDR3 runs from 105->116 and fw4 excluded).
        Pass True if rearranged V(D)J sequence of for_gapped_seq is True
        (CDR3 runs from 105->117 and fw4 is included).
        See http://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html

    for_gapped_seq : bool, optional
        True to return the string index positions (0-based inclusive)
        according to those used for creating a gapped sequence of fixed size. 
        See get_imgt_residue_names(). By default, False (original IMGT numbering).

    Returns
    -------
    Union[Dict, Tuple]
        as_dict=False: A tuple containing tuples of region name, from and to residue
            positions (1-based inclusive or 0-based inclusive if for_gapped_seq is True).

        as_dict=True: A dictionary where key is the region name and values are
            a tuple containing the from and to residue positions
            (1-based inclusive or 0-based inclusive if for_gapped_seq is True).
    """

    if for_gapped_seq and not is_rearranged:
        raise Exception(
            'If for_gapped_seq is True, is_rearranged must also be True.')

    # IMGT region positions from:
    # http://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html
    # Must be in position ascending order (see for_gapped_seq below.)
    # Positions are 1-based inclusive.
    regions = [
        (f'fw{chain}1', 1, 26),
        (f'cdr{chain}1', 27, 38),
        (f'fw{chain}2', 39, 55),
        (f'cdr{chain}2', 56, 65),
        (f'fw{chain}3', 66, 104),
        # TODO: Possibility of missed 117?
        (f'cdr{chain}3', 105, 117 if is_rearranged else 116)
    ]
    if is_rearranged:
        regions.append((f'fw{chain}4', 118, 129))
    REGIONS = tuple(regions)

    # For gapped sequences, this gets the from-to character indexes
    # for each region. 0-based inclusive.
    if for_gapped_seq:

        # Get the list of IMGT residue names, but in a tuple.
        # The tuple index acts as the residue index in the gapped sequence.
        imgt_residue_names = get_imgt_residue_names(as_dict=False)

        # Init
        i = 0
        regions_for_gapped_seq = []
        curr_region_index = 0
        gapped_region_index_start = 0

        # Loop through the list of residue names that can appear
        # in a gapped sequence.
        for i, imgt_pos_name in enumerate(imgt_residue_names):

            # Get the imgt region
            curr_region_name, curr_region_from, curr_region_to = REGIONS[curr_region_index]

            # Remove insertion letter if exists
            # (e.g. 111A -> 111) and convert to int.
            imgt_pos_name2 = int(
                imgt_pos_name[:-1]) if not imgt_pos_name[-1].isnumeric() else int(imgt_pos_name)

            # Add new gapped seq region if at end of the current IMGT region.
            if imgt_pos_name2 == curr_region_to:
                regions_for_gapped_seq.append(
                    (
                        curr_region_name,
                        gapped_region_index_start,
                        i  # Region end (inclusive)
                    )
                )
                gapped_region_index_start = i + 1

                # Move to the next imgt region
                curr_region_index += 1
                if curr_region_index > len(REGIONS):
                    raise Exception(
                        f"Recieved higher IMGT number than expected: '{imgt_pos_name}'"
                    )

    region_list = REGIONS if not for_gapped_seq else regions_for_gapped_seq
    if as_dict:

        return {region: (pos_from, pos_to) for region, pos_from, pos_to in region_list}
    else:
        return region_list


def _parse_v_germline_to_imgt(seq: str, chain: str, remove_gaps=True) -> dict:
    """Applies IMGT annotation to V-REGION germline gene sequence

    Args:
        seq (str): Gapped IMGT V-REGION germline amino acid sequence
            e.g. 'YTF....TSY'

        chain (str): chain letter, e.g. (h)eavy, (a)lpha

        remove_gaps: If True, gaps ('.') are removed from the numbered
            sequences. Required when comparing to OAS data units which
            have the gaps removed.


    Returns:
        dict: Annotated sequence compatible with OAS Data Unit 'data' property,
          e.g. {'cdrh1': {'27': 'Y', '28': 'T' ...}}

    Links:
        IMGT definition: http://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html

    """

    # IMGT region positions from:
    # http://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html
    regions = get_region_positions(
        chain=chain, as_dict=False, is_rearranged=False)

    # Build annotations for each region
    seq_annotated = {}
    for r in regions:
        r_name = r[0]
        r_from = r[1]
        r_to = r[2]

        seq_extract = list(seq[r_from - 1:r_to])  # aas, -1 as zero based
        # imgt numbered positions, use str for OAS compatibility
        seq_pos = map(str, range(r_from, r_to + 1))
        seq_annotated[r_name] = dict(zip(seq_pos, seq_extract))

        # Remove gapped residues
        # (Perform this after sequential numbering above)
        if remove_gaps:
            seq_annotated[r_name] = {pos: residue
                                     for pos, residue in seq_annotated[r_name].items()
                                     if residue != '.'}

    return seq_annotated

 # ---- PARSING END -----


def build_distance_matrix(df: pd.DataFrame, distance_function, measure_value='aa_annotated',
                          triangle_inequality_samples=10000, **kwargs) -> pd.DataFrame:
    """Build distance matrix from IMGT references

    Args:
        df (pd.DataFrame): IMGT reference sequences, translated and annotated.

        distance_function (function): Distance function that takes two items and 
            returns a single distance metric.

        measure_value (str, optional): Column name containing annotated amino 
            acid sequences. Defaults to 'aa_annotated'.

        triangle_inequality_samples (int, optional): A random sample of distances 
            will be tested to ensure the triangle inequality holds. Defaults to 10000.

    Raises:
        Exception: If triangle inequality does not hold.

    Returns:
        pd.DataFrame: Square distance matrix, headers will be same as df.index.
    """

    # Create the distance matrix
    df_distances = distance.create_distance_matrix(df,
                                                   distance_function, measure_value, **kwargs)

    # Check for breach of triangle inequality
    triangle_test = distance.verify_triangle_inequality(df_distances,
                                                        triangle_inequality_samples)
    if len(triangle_test) > 0:
        print(triangle_test)
        raise Exception('Triangle inequality does not hold')

    # Return
    return df_distances


def _coord_distances(df_coords: pd.DataFrame, df_orig_dist: pd.DataFrame) -> pd.DataFrame:
    """Create square matrix of plotted distances

    Args:
        df_coords (DataFrame): Computed xy coordinates for each sequence.

        df_orig_dist (DataFrame): The original distance matrix (just used to
            get the list of row and column headers).

    Returns:
        pd.DataFrame: A square matrix with Euclidean distances between
            plotted points.
    """

    # Euclidean distance between plotted points.
    # Condensed distance matrix.
    plot_dist_matrix = pdist(df_coords[['x', 'y']])

    # Condensed to Square + add headers
    plot_dist_matrix = squareform(plot_dist_matrix)
    assert (plot_dist_matrix.shape == df_orig_dist.shape)

    # DataFrame with index and column headers
    df_plot_dist = pd.DataFrame(plot_dist_matrix,
                                index=list(df_orig_dist.index),
                                columns=list(df_orig_dist.columns))
    assert (df_plot_dist.shape == df_orig_dist.shape)

    # Return
    return df_plot_dist


def _coord_melt_distances(df_orig_dist: pd.DataFrame,
                          df_plot_dist: pd.DataFrame,
                          df_dist_delta: pd.DataFrame) -> pd.DataFrame:
    """Create long table of original vs plotted distances

    Args:
        df_orig_dist (pd.DataFrame): Original pairwise computed distances
        df_plot_dist (pd.DataFrame): Plotted pairwise distances
        df_dist_delta (pd.DataFrame): Difference between original and plotted distances

    Returns:
        pd.DataFrame: Long DataFrame combining data from all DataFrames
    """

    # Convert to tall tables
    df_orig_dist_melt = df_orig_dist.reset_index().melt(id_vars='index',
                                                        var_name='item2', value_name='orig_distance')
    df_orig_dist_melt.set_index(['index', 'item2'], inplace=True)

    df_plot_dist_melt = df_plot_dist.reset_index().melt(id_vars='index',
                                                        var_name='item2', value_name='plot_distance')
    df_plot_dist_melt.set_index(['index', 'item2'], inplace=True)

    df_dist_delta_melt = df_dist_delta.reset_index().melt(id_vars='index',
                                                          var_name='item2', value_name='dist_delta')
    df_dist_delta_melt.set_index(['index', 'item2'], inplace=True)

    # Join all three by item1 and item2 indexes
    df_melted = df_orig_dist_melt.join(
        df_plot_dist_melt, how='left', sort=True)
    df_melted = df_melted.join(df_dist_delta_melt, how='left', sort=True)
    df_melted.rename(columns={'index': 'item1'},
                     inplace=True)  # rename index -> item1

    # Return
    return df_melted


def compute_coords(df_dist_matrix: pd.DataFrame,
                   method: str,
                   random_state: int,
                   n_neighbors: int = 15):
    """Computes anchor coordinates from distance matrix

    Args:
        df_dist_matrix (DataFrame): Square distance matrix.

        method (str): Dimensionality reduction method. Valid values are
            'MDS', 'TSNE', 'PCA' or 'UMAP'.

        random_state (int): Random state (int to fix, None for random).

        n_neighbors (int): UMAP only, number of closest neighbours.

    Raises:
        Exception: If invalid value is supplied for method.

    Returns:
        Tuple[DataFrame, DataFrame]: Coordinates [0] and DataFrame 
            with original distances, plotted distances and difference between the two
    """

    # Fit and compute coordinates
    coords = None
    if method == 'PCA':
        model = PCA(n_components=2, random_state=random_state)
        coords = model.fit_transform(df_dist_matrix)
    elif method == 'TSNE':
        model = manifold.TSNE(n_components=2, random_state=random_state)
        coords = model.fit_transform(df_dist_matrix)
    elif method == 'MDS':
        model = manifold.MDS(n_components=2,
                             dissimilarity="precomputed",
                             random_state=random_state)
        coords = model.fit(df_dist_matrix).embedding_  # [[x,y]] coordinates
    elif method == 'UMAP':
        import umap  # Potentially slow to import due to Numba compile?
        model = umap.UMAP(n_neighbors=n_neighbors,
                          n_components=2,
                          random_state=random_state)
        coords = model.fit_transform(df_dist_matrix)

    else:
        raise Exception(f"Invalid method: '{method}'")

    # Dataframe of coordinates
    df_coords = pd.DataFrame(coords,
                             index=list(df_dist_matrix.index),
                             columns=['x', 'y'])
    df_coords.index.name = 'index'

    # DataFrame of plotted distances
    df_plot_dist = _coord_distances(df_coords, df_dist_matrix)
    df_plot_dist.index.name = 'index'

    # Actual vs Plotted distances
    df_dist_delta = df_plot_dist - df_dist_matrix

    # Combine all 3 into one long table
    df_dist_melted = _coord_melt_distances(df_orig_dist=df_dist_matrix,
                                           df_plot_dist=df_plot_dist,
                                           df_dist_delta=df_dist_delta)

    # Return coords and distances
    return df_coords, df_dist_melted


def plot_anchor_distortion(df_melted: pd.DataFrame) -> Figure:
    """Generate anchor distortion plots.

    Args:
        df (DataFrame): Original and plotted distances (df_melted)

    Returns:
        Tuple(Figure, Figure): Plotly figures (groups, overall)
    """

    # Init
    df = df_melted.copy()
    df = df.reset_index()

    # Get groups (IGHV1 etc.)
    df['group_name'] = [x[:5]
                        for x in list(df['index'])]
    df['group_name2'] = [x[:5]
                         for x in list(df['item2'])]

    is_same_group = df['group_name'] == df['group_name2']

    # Plot distortion within same groups
    df_same_group = df[is_same_group]
    fig1: Figure = px.density_contour(df_same_group, facet_col_wrap=3,
                                      x='orig_distance', y='plot_distance', facet_col='group_name')

    fig1.update_traces(contours_coloring="fill", contours_showlabels=False)
    fig1.update_traces(showscale=False)

    # Plot overall distortion
    fig2: Figure = px.density_contour(
        df, x='orig_distance', y='plot_distance')
    fig2.update_traces(contours_coloring="fill", contours_showlabels=True)

    # Return the figures
    return fig1, fig2


# %% Plot anchors
def plot_anchor_positions(df_coords: pd.DataFrame) -> Figure:
    """Scatter plot of anchor coordinates

    Args:
        df_coords (pd.DataFrame): Anchor coordinates. Index should be the name of the allele.
            First 5 characters will be used for group (e.g. IGHV1)

    Returns:
        Figure: Plotly scatter figure, coloured by group.
    """

    df = df_coords.copy()
    df = df.reset_index()

    # group by IGHV1 etc.
    df['group_name'] = [x[:5] for x in list(df['index'])]
    fig = px.scatter(x=df['x'], y=df['y'],
                     color=df['group_name'])

    return fig

# %%


def db_write_records(df_records: pd.DataFrame, fn_db: str):
    """Create anchor_records table

    Existing table will be overwritten. Creates a unique
    index on anchor_id and anchor_name fields.

    Args:
        df_records (pd.DataFrame): DataFrame containing the anchor records.
        fn_db (str): Database to write to.
    """

    sql_index1 = '''
        CREATE UNIQUE INDEX idx_records_anchor_id
        ON anchor_records(anchor_id)
    '''

    sql_index2 = """
        CREATE UNIQUE INDEX idx_records_anchor_name
        ON anchor_records(anchor_name)
    """

    # Create table and index
    with sqlite3.connect(fn_db) as conn:
        df_records = df_records.copy().reset_index()
        df_records.rename(columns={'index': 'anchor_name'},
                          inplace=True)  # index column is loaded from csv
        # df_records = df_records.convert_dtypes() # Convert from object type
        df_records.to_sql('anchor_records', con=conn, if_exists='replace',
                          index=True, index_label='anchor_id')
        curr = conn.cursor()
        curr.execute(sql_index1)
        curr.execute(sql_index2)
        curr.close()


def db_write_coords(df_coords: pd.DataFrame, fn_db: str):
    """Create database of anchors

    Args:
        df_coords (pd.DataFrame): Anchor coordinates. 'index' column should be the name of the allele.

        fn_db (str): SQLite db to create/write to. 'anchor_records' should exist as anchor_id will
            be pulled from here.
    """

    TABLE_NAME = 'anchor_coords'

    sql_index_anchor_id = f"""
        CREATE UNIQUE INDEX idx_coords_anchor_id
        ON {TABLE_NAME}(anchor_id)
    """

    sql_index_anchor_name = f"""
        CREATE UNIQUE INDEX idx_coords_anchor_name
        ON {TABLE_NAME}(anchor_name)
    """

    # Update anchor_id to be that defined by anchor_records
    sql_update_anchor_id = f"""
        UPDATE {TABLE_NAME}
        SET anchor_id = (SELECT anchor_id
                         FROM anchor_records
                         WHERE anchor_name = anchor_coords.anchor_name)
    """

    with sqlite3.connect(fn_db) as conn:
        curr = conn.cursor()

        # Prepare DataFrame
        df_coords = df_coords.copy()  # Stop changes affecting
        df_coords.insert(0, 'anchor_id', 0)  # will update shortly...
        df_coords.rename(columns={'index': 'anchor_name'},
                         inplace=True)  # index column is loaded from csv

        # Create anchor_coords table
        df_coords.to_sql(name=TABLE_NAME, con=conn, if_exists='replace',
                         index=False)

        # Update anchor_id and index
        curr.execute(sql_update_anchor_id)
        curr.execute(sql_index_anchor_id)
        curr.execute(sql_index_anchor_name)
        curr.close()


def get_imgt_residue_names(as_dict=False) -> Union[Dict, Tuple]:
    """
    Returns a tuple of all IMGT-numbered positions that are likely
    to be encountered in OAS datasets.

    IMGT numbering is per the ANARCI numbered sequences in OAS.
    CDR3 insertions are represented as letters between 111 and 112
    symmetrically, e.g. '111A'->'111B'->'112B'->'112A'.

    This function can be used to generate gapped sequences of
    fixed length, by mapping IMGT residue names to the index of the
    element in the returned tuple.

    NOTE: Consider caching the result if using in a loop.

    Parameters
    ----------
    as_dict : bool, optional
        True to return a dictionary or False to return a tuple of 
        residue names. By default False.

    Returns
    -------
    Tuple (as_dict=False)
        A tuple of IMGT residue names (per those in OAS).

    Dict (as_dict=True)
        A dictionary where keys are the IMGT residue names and
        values are the index position.

    """

    # IMGT region positions from:
    # http://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html

    # From fw1 to middle of cdr3, 1 to 112, just numbered residues (no letters)
    residue_names = [str(i) for i in range(
        1,
        112  # exclusive (111)
    )]

    # Add CDR3 insertions.
    # CDR3 insertions in OAS are inserted as letters symmetrically around residues
    # 111 and 112. E.g. 111 -> 111A -> 111B -> 112B -> 112A -> 112...
    # The AHo numbering system (which doesn't use letters for insertions)
    # allows up to 35 residues for CDR 3 (in Aho numbering, from 106 to 140, centered around 123),
    # and should accomodate most natural occurences.
    # We allow 36, 111 -> 111Q, 112Q -> 112.
    from_char, to_char = 'A', 'Q'
    cdr3_insertions = [
        # +1 as excl.
        '111' + chr(i) for i in range(ord(from_char), ord(to_char) + 1, 1)
    ]
    cdr3_insertions.extend(
        ['112' + chr(i)
         for i in range(ord(to_char), ord(from_char) - 1, -1)]  # -1, -1 as excl and in reverse.
    )
    residue_names.extend(cdr3_insertions)

    # Remaining cdr3 from 112 to fw4 129.
    residue_names.extend(
        [str(i) for i in range(
            112,
            130  # exclusive (129)
        )]
    )

    # Return
    if not as_dict:
        return tuple(residue_names)
    else:
        return {v: i for i, v in enumerate(residue_names)}


def imgt_to_gapped(imgt_seq: Dict, imgt_residue_names: Dict, gap_char: str = '.') -> str:
    """
    Convert IMGT-numbered sequence to a fixed-length gapped string.

    Parameters
    ----------
    imgt_seq : Dict
        The IMGT-numbered sequence from an OAS dataset (ANARCI-numbered).
        e.g. {'fwh1':{'24':'A', '25':'A'... }}

    imgt_residue_names : Dict
        Dictionary to map IMGT-numbered residues to an index position.
        e.g. {'111A': 112...}

    gap_char : str
        The character to use for gaps in the gapped sequence.

    Returns
    -------
    str
        The gapped sequence, e.g. 'AA...GV..'. This will be the length of
        [imgt_residue_names].
    """

    # Init fixed-length character array
    seq_residues = list(gap_char * len(imgt_residue_names))

    # Set the characters
    for region in imgt_seq.values():
        for imgt_pos, residue in region.items():
            index_pos = imgt_residue_names[imgt_pos]
            seq_residues[index_pos] = residue

    # Join and return
    return ''.join(seq_residues)
