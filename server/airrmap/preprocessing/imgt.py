# Functionality for processing of IMGT VDJ reference sequences
# (anchor sequences)

# tests > test_imgt.py

# %% Imports
import pandas as pd
from Bio import SeqIO
from typing import Dict, Tuple, Union

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
