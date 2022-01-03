# Show how position changes with morphed sequences (CDRH1-CDRH2 only).
# Sets random pair of IMGT reference V gene sequences.

# Use (see bottom of script):
# 1. Set seq_from_name and seq_to_name (loads gapped aa sequence from anchors db), or None for random.
# 2. Set env_name (with CDRH1-H2 anchors).
# 3. Run all.
# 4. To copy sequences on mac for presentation, first copy to Notes to keep line breaks, then Excel/PowerPoint.
# 5. Can drag plot image to PowerPoint (copy doesn't appear to work).

# > Tests (bottom)

# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import seaborn as sns
import json
import plotly.express as px
import plotly.graph_objects as go
import unittest
from typing import Dict, List, Optional, Sequence, Tuple

import airrmap.shared.analysis_helper as ah
import airrmap.preprocessing.compute_selected_coords as compute_coords
import airrmap.preprocessing.imgt as imgt
from airrmap.shared.models import RepoQuery


# %% Get the sequence regions
def get_seq_regions(v_seq: str, seq_regions: List[str], chain: str = 'h'):
    """
    Takes an IMGT-gapped V germline gene sequence (a.a.) and returns IMGT-numbered sequences.

    Parameters
    ----------
    v_seq : str
        The IMGT-gapped V germline gene sequence (amino acids).
        e.g. 'QVQLVQSGA.EVKKPGASVKVSCKASGYTF....TS...'

    seq_regions : List[str]
        The regions to extract (e.g. ['cdrh1', 'cdrh2']).

    chain : str, optional
        The chain letter, used in seq_regions (e.g. 'h' for 'heavy'.). By default 'h'.

    Returns
    -------
    Dict
        The gapped string sequences for each region, e.g. {'cdrh1': 'GGSIS..SGDYY'}
    Dict
        The numbered regions -without- gaps, e.g. {'cdrh1': {'56': 'I', ...}}
    """

    # Convert to numbered with gaps removed
    imgt_numbered_seq_ungapped = imgt._parse_v_germline_to_imgt(
        seq=v_seq,
        chain=chain,
        remove_gaps=True
    )

    # Convert to numbered with gaps
    imgt_numbered_seq_gapped = imgt._parse_v_germline_to_imgt(
        seq=v_seq,
        chain=chain,
        remove_gaps=False
    )

    # Extract the regions
    # Example: {'cdrh1': {'56' : 'I'...}}
    result_numbered: Dict = {}
    result_str: Dict = {}
    for region in seq_regions:
        # ungapped for numbered seq
        result_numbered[region] = imgt_numbered_seq_ungapped[region]
        result_str[region] = ''.join(
            [residue for position, residue in imgt_numbered_seq_gapped[region].items()])

    # Return
    return result_str, result_numbered

# %% Get mutations


def get_mutations(seq1: str, seq2: str):
    """
    Build list of mutations to transform seq1 into seq2.

    Both sequences must be the same length. Scans
    left to right, comparing each character.
    If characters differ, a new mutation is added to the list.

    Parameters
    ----------
    seq1 : str
        First sequence, e.g. 'AAA'
    seq2 : str
        Second sequence which must be same length as seq1, e.g. 'CCC'

    Returns
    -------
    List[Tuple]
        List of tuples where each tuple contains
        the zero-based character index, the original character/residue,
        and the new character/residue.
        e.g. 
        'AAA' -> 'AAC'
        [(2, 'A', 'C)]

    Raises
    ------
    ValueError
        If sequences are the not the same length.
    """

    # Check lengths are same
    if len(seq1) != len(seq2):
        raise ValueError('Sequences must be the same length.')

    # Build list of single-residue mutations
    length = len(seq1)
    mutations = []
    for i in range(length):
        if seq1[i] != seq2[i]:
            mutations.append((i, seq1[i], seq2[i]))

    # Return
    return mutations

# %% Apply mutation


def apply_mutation(seq: str, mutation: Tuple[int, str, str]) -> str:
    """
    Apply a single mutation to a sequence.

    Parameters
    ----------
    seq : str
        [description]
    mutation : Tuple[int, str, str]
        The mutation to apply. A tuple with the following values:
            0: zero-based character position in seq.
            1: the original character/residue.
            2: the mutated character/residue.

    Returns
    -------
    str
        The mutated sequence.

    Raises
    ------
    Exception
        If the residue at the mutating position in seq is not the same
        as the original residue provided in the mutation.
    """

    seq_list = list(seq)
    position, from_char, to_char = mutation

    # Verify from_char is as epxected
    if seq_list[position] != from_char:
        msg = f"Unexpected character in position {position}, " + \
            f"expecting {from_char} but got {seq_list[position]}. Seq is '{seq}'."
        raise Exception(msg)

    # Apply the mutation
    seq_list[position] = to_char

    # Return
    return ''.join(seq_list)

# Gapped to IMGT


def gapped_cdrh1h2_to_imgt(gapped_seq: str) -> Dict:
    """
    Take a gapped CDRH1-H2 sequence and return the IMGT-numbered equivalent.

    Parameters
    ----------
    gapped_seq : str
        The gapped CDRH1-H2 concatenated sequence.

    Returns
    -------
    Dict
        IMGT-numbered sequence for CDRH1 and CDRH2 only.
        e.g. {'cdrh1': {'27': 'G', '28': 'S' ... }}
    """

    # Used to map gapped CDRH1-H2 string back to IMGT numbered format
    # IMGT region positions from:
    # http://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html
    map_to_imgt = (
        ('27', 'cdrh1'),
        ('28', 'cdrh1'),
        ('29', 'cdrh1'),
        ('30', 'cdrh1'),
        ('31', 'cdrh1'),
        ('32', 'cdrh1'),
        ('33', 'cdrh1'),
        ('34', 'cdrh1'),
        ('35', 'cdrh1'),
        ('36', 'cdrh1'),
        ('37', 'cdrh1'),
        ('38', 'cdrh1'),
        ('56', 'cdrh2'),
        ('57', 'cdrh2'),
        ('58', 'cdrh2'),
        ('59', 'cdrh2'),
        ('60', 'cdrh2'),
        ('61', 'cdrh2'),
        ('62', 'cdrh2'),
        ('63', 'cdrh2'),
        ('64', 'cdrh2'),
        ('65', 'cdrh2')
    )

    # Take gapped strings and construct
    # imgt-numbered dictionary.
    # (CDRH1-H2 only).
    # Must be same length.
    # (will raise error otherwise)
    seq_imgt: Dict = {}

    # Loop through mappings
    for i in range(len(map_to_imgt)):
        map_item = map_to_imgt[i]
        pos_name, region_name = map_item
        if region_name not in seq_imgt:
            seq_imgt[region_name] = {}

        # Set the residue (if not a gap)
        if gapped_seq[i] != '.':
            seq_imgt[region_name][pos_name] = gapped_seq[i]

    # Return numbered dict
    return seq_imgt


def main(env_name: str, seq_from_name: Optional[str] = None, seq_to_name: Optional[str] = None):
    """
    Generate plot of sequence positions morphing from/to.

    Loads the sequences from the anchor names, which should be of equal length
    (CDR1 / CDR2 sequences only). Generates and applies a list of single-residue
    mutations from left to right, such that the 'from' sequence is mutated
    into the 'to' sequence.

    Parameters
    ----------
    env_name : str
        The environment name.

    seq_from_name : str, optional
        The name of the 'from' anchor sequence, e.g. IGHV1-18*01.
        If None, will use a random anchor. By default, None.

    seq_to_name : str, optional
        The name of the 'to' anchor sequence, e.g. IGHV3-15*06.
        If None, will use a random anchor. By default, None.

    Raises
    ------
    Exception
        If mutations applied to the 'from' sequence don't result
        in the 'to' sequence.
    """

    # Field in the anchor db containing the gapped aa sequences.
    GAPPED_AA_FIELD = 'aa'
    ANCHOR_NAME_FIELD = 'anchor_name'

    # %% Init
    repo, appcfg = ah.get_repo_cfg()
    envcfg = appcfg.get_env_config(env_name)
    seqcfg = envcfg['sequence']
    anccfg = envcfg['anchor']
    seq_regions = envcfg['distance_measure_options']['regions']  # e.g. cdrh1, cdrh2

    # %% Get query
    query = RepoQuery(env_name,
                      value1_field='redundancy',
                      value2_field='seq'
                      )

    # %% Get data
    df_seq, report = repo.run_coords_query(appcfg, query.to_dict())

    # %% Load anchors, add subgroup and get aa gapped seq from anchor name
    df_anchors = ah.load_anchors(env_name)
    df_anchors['subgroup'] = df_anchors['anchor_name'].apply(lambda x: x[:5])

    # %% Get the sequences
    # Select 2 at random to use if seq_from_name or seq_to_name is None.
    df_anchors_sample = df_anchors.sample(n=2, replace=False)
    if seq_from_name is None:
        seq_from = str(df_anchors_sample[GAPPED_AA_FIELD].iloc[0])
        seq_from_name = df_anchors_sample[ANCHOR_NAME_FIELD].iloc[0]
    else:
        seq_from = str(df_anchors[df_anchors['anchor_name']
                                  == seq_from_name]['aa'].iloc[0])

    if seq_to_name is None:
        seq_to = str(df_anchors_sample[GAPPED_AA_FIELD].iloc[1])
        seq_to_name = str(df_anchors_sample[ANCHOR_NAME_FIELD].iloc[1])
    else:
        seq_to = str(df_anchors[df_anchors['anchor_name']
                                == seq_to_name]['aa'].iloc[0])

    # %% Morph sequences
    # Region positions: (should match imgt._parse_v_germline_to_imgt()).
    # fwh1 (1>), cdrh1 (27>), fwh2 (39>), cdrh2 (56>), fwh3 (66>).

    # |fwh1                     |cdrh1      |fwh2            |cdrh2    |fwh3
    # 1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123
    #          1         2         3         4         5         6         7         8         9         1
    # QVQLQDSGP.GLVKPSQTLSLTCTVSGGSIS..SGDYYWSWIRQPPGKGLEWIGYFYYS...GSTYYNPSLK.SRVTISVDTSKNQFSLKLSSVTAADTAVYY

    # %% Generate morph sequences
    seq_from_str, seq_from_numbered = get_seq_regions(
        v_seq=seq_from, seq_regions=seq_regions, chain='h')
    seq_from_str_concat = ''.join(
        [seq for region, seq in seq_from_str.items()])
    seq_to_str, seq_to_numbered = get_seq_regions(
        v_seq=seq_to, seq_regions=seq_regions, chain='h')
    seq_to_str_concat = ''.join([seq for region, seq in seq_to_str.items()])

    # %% Build list of mutations
    mutation_list = get_mutations(
        seq1=seq_from_str_concat, seq2=seq_to_str_concat)
    morph_seq_list = [seq_from_str_concat]
    last_seq = seq_from_str_concat
    for mutation in mutation_list:
        mutated_seq = apply_mutation(last_seq, mutation)
        morph_seq_list.append(mutated_seq)
        last_seq = mutated_seq

    # Verify we ended up in the right place
    if last_seq != seq_to_str_concat:
        raise Exception(
            f"Mutations don't result in expected sequence, expected '{seq_to_str_concat}', got '{last_seq}')")

    # %% IMGT number the morph sequences
    morph_seqs_numbered = [gapped_cdrh1h2_to_imgt(x) for x in morph_seq_list]

    # %% Compute the coordinates (uses regions defined in environment)
    morph_coords_list = compute_coords.get_coords(
        env_name=env_name,
        seq_list=[json.dumps(x) for x in morph_seqs_numbered],
        app_cfg=appcfg
    )

    # %% Init colour mappping
    col_norm = mpl.colors.Normalize(vmin=0, vmax=len(morph_coords_list) - 1)
    cmap = cm.hsv
    #cmap = 'turbo'
    col_scaler = cm.ScalarMappable(norm=col_norm, cmap=cmap)

    # %% Plot anchors and morph seqs.
    axes = df_anchors.plot.scatter(
        x='x', y='y', c='#cecece', marker='.', figsize=(10, 7))
    for i, seq_coords in enumerate(morph_coords_list):
        marker_type = '>' if (i != 0 and i != len(
            morph_coords_list) - 1) else 'o'
        x = seq_coords['sys_coords_x']
        y = seq_coords['sys_coords_y']

        marker_colour = col_scaler.to_rgba(i)
        axes.plot(x, y, c=marker_colour, marker=marker_type, linestyle='none')

        if i > 0:
            prev_seq_coords = morph_coords_list[i - 1]
            x_prev = prev_seq_coords['sys_coords_x']
            y_prev = prev_seq_coords['sys_coords_y']
            arrow = mpl.patches.FancyArrowPatch(
                posA=(x_prev, y_prev),
                posB=(x, y),
                arrowstyle='simple',
                # arrowstyle='-',
                mutation_scale=20,
                color=marker_colour
                # shrinkA=50,
                # shrinkB=50
            )
            axes.add_patch(arrow)

    axes.set(aspect='equal')

    # %% Show sequences
    print(seq_from_name)
    for i, morph_seq in enumerate(morph_seq_list):
        num_item = str(i+1) + '.'
        num_item = num_item.ljust(3)
        print(f'{num_item} {morph_seq}')
    print(seq_to_name)


# %% Run
#main('Anchor_MLAT_Analysis', seq_from_name='IGHV2-70*19', seq_to_name='IGHV4-59*13')
main('ENV_NAME', seq_from_name=None, seq_to_name=None)


# %% 
#env_name = 'ENV_NAME'
#df_anchors = ah.load_anchors(env_name)
#px.scatter(df_anchors, x='x', y='y', hover_name='anchor_name')




# ---- TESTS ----

# %%


class TestSeqMorph(unittest.TestCase):

    def test_get_mutations(self):
        seq1 = 'ABC'
        seq2 = 'CCC'
        mutations = get_mutations(seq1, seq2)
        self.assertListEqual(
            mutations,
            [(0, 'A', 'C'), (1, 'B', 'C')],
            'Correct list of mutations should be returned.'
        )

    def test_get_seq_regions(self):
        seq = 'QVQLQDSGP.GLVKPSQTLSLTCTVSGGSIS..SGDYYWSWIRQPPGKGLEWIGYFYYS...GSTYYNPSLK.SRVTISVDTSKNQFSLKLSSVTAADTAVYY'  # IGHV4-30-4*04
        result_str, result_numbered = get_seq_regions(
            v_seq=seq,
            seq_regions=['cdrh1', 'cdrh2'],
            chain='h'
        )

        self.assertDictEqual(
            result_str,
            {'cdrh1': 'GGSIS..SGDYY', 'cdrh2': 'FYYS...GST'},
            'Gapped string sequences for selected regions should be returned.'
        )

        self.assertDictEqual(
            result_numbered,
            {'cdrh1': {'27': 'G',
                       '28': 'G',
                       '29': 'S',
                       '30': 'I',
                       '31': 'S',
                       '34': 'S',
                       '35': 'G',
                       '36': 'D',
                       '37': 'Y',
                       '38': 'Y'},
             'cdrh2': {'56': 'F',
                       '57': 'Y',
                       '58': 'Y',
                       '59': 'S',
                       '63': 'G',
                       '64': 'S',
                       '65': 'T'}},
            'Dictionary of ungapped numbered residues for selected regions should be returned.'
        )

    def test_apply_mutation(self):
        seq1 = 'AC'
        seq2_actual = 'CC'
        mutation = (0, 'A', 'C')
        seq2_test = apply_mutation(seq1, mutation)

        self.assertEqual(
            seq2_test,
            seq2_actual,
            'Applying mutation to seq1 should result in seq2.'
        )

    def test_gapped_cdrh1h2_to_imgt(self):
        gapped_seq = 'GGSIS..SGDYYFYYS...GST'
        expected_result = {'cdrh1': {'27': 'G',
                                     '28': 'G',
                                     '29': 'S',
                                     '30': 'I',
                                     '31': 'S',
                                     '34': 'S',
                                     '35': 'G',
                                     '36': 'D',
                                     '37': 'Y',
                                     '38': 'Y'},
                           'cdrh2': {'56': 'F',
                                     '57': 'Y',
                                     '58': 'Y',
                                     '59': 'S',
                                     '63': 'G',
                                     '64': 'S',
                                     '65': 'T'}}

        imgt_numbered = gapped_cdrh1h2_to_imgt(gapped_seq)
        self.assertDictEqual(
            imgt_numbered,
            expected_result,
            'Correct IMGT-numbered dictionary should be returned for gapped sequence, and gaps should not be included.'
        )


# %%
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
# %%
