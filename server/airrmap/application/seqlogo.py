# Create sequence logos from gapped sequences.

# %% Imports
import pandas as pd
import io
import json
import re
import numpy as np
import logomaker
import base64
from typing import Any, List, Optional, Union
from numba import jit, njit, prange

import airrmap.preprocessing.imgt as imgt
import airrmap.shared.analysis_helper as ah
from airrmap.shared.models import RepoQuery
from airrmap.application.config import AppConfig, SeqFileType


# TODO: parallel=True seems slower during testing...
@njit(parallel=False)
def count_residues(A, weights):
    """
    Fast frequency count of residues at each residue position.

    Parameters
    ----------
    A : ndarray
        A numpy matrix in the shape of (sequence count, sequence length).
        Values should be ascii codes of each residue.

        For example:
        Original sequences:
        ['AC','AY'] ->

        Transformed to matrix of ascii codes:
        [[65, 67]
         [65, 89]]

    weights : ndarray
        A 1D numpy array containing weights for each sequence 
        (e.g. redundancy). This will be applied to the frequency
        counts.

    Returns
    -------
    ndarray
        A numpy matrix in the shape of (sequence_length, 256), where
        values represent the frequency of each character/ascii code at
        each position in a sequence.
    """

    # Fixed sequence length
    SEQ_COUNT = int(A.shape[0])
    SEQ_LENGTH = A.shape[1]
    ASCII_CHARS = 256

    # Check weights shape is same as sequence
    if weights.shape[0] != SEQ_COUNT:
        raise Exception('Differing number of weights and sequences.')

    # Per position, 256 chars (ascii counts
    ctrs = np.zeros((SEQ_LENGTH, ASCII_CHARS))

    # For each sequence and residue position,
    # increment the counter for that position and ascii code.
    # Without "parallel=True" in the jit-decorator
    # the prange statement is equivalent to range
    for seq_index in prange(SEQ_COUNT):
        for char_index in prange(SEQ_LENGTH):
            ascii_code = A[seq_index, char_index]
            ctrs[char_index, ascii_code] += weights[seq_index]
    return ctrs


def get_logo(gapped_seqs: np.ndarray,
             weights: Optional[np.ndarray] = None,
             title: str = '',
             encode_base64: bool = False,
             image_format='png'):
    """
    Create a sequence logo from a gapped sequence.

    Counts the frequency of each residue type at each
    sequence position.

    Parameters
    ----------
    gapped_seqs : np.ndarray
        1D numpy array of gapped sequences (str). 
        Gaps should be '.' character.

    weights : np.ndarray[int], optional
        1D numpy array of weights to apply to 
        frequency counts, e.g. sequence redundancy values.
        Should be the same size as the gapped_seqs array.
        If None, weights will be set to an array of ones.

    title : str, optional
        Title for the logo, by default ''.

    encode_base64 : bool, optional
        True to base64-encode the logo image. By default False.

    image_format : str, optional
        Image format for the logo image if encode_base64 is True.
        By default 'png'.

    Returns
    -------
    logomaker.Logo or str: 
        The sequence logo (a matplotlib plot) or the base64-encoded image(utf-8).

    DataFrame: 
        The frequency counts used for the logo.
    """

    # %% Get list of sequences
    num_seqs = gapped_seqs.shape[0]
    fixed_seq_length = len(gapped_seqs[0])  # use first seq to get length

    # Prepare sequences for counting.
    # Fast conversion of string sequences
    # to arrays of ascii codes for improved counting performance.
    # 1. .astype(bytes) converts strs to byte strings ('AB' -> b'AB')
    # 2. frombuffer(, dtype='uint8') converts byte strings to ascii arrays
    #     (b'AB' -> [65, 66]), but will be flat array.
    # 3. Reshape converts flat array to matrix (100 -> [10, 10]) where
    #    rows=sequences and columns=ascii codes.
    seq_list = np.frombuffer(
        gapped_seqs.astype(bytes),
        dtype='uint8'
    ).reshape((num_seqs, fixed_seq_length))

    # Get weights
    if weights is None:
        weights_list = np.ones(num_seqs).astype(np.uint32)
    else:
        weights_list = weights.astype(np.uint32)

    # %% Count
    #seq_ctrs = count_gapped_chars(seq_list, weight_list)
    seq_ctrs = count_residues(seq_list, weights_list)

    # %% Select only valid chars (i.e. selected columns from the counts (0-256))
    # Must be in ascii order due to counts array.
    # Removed B, U, X, Z, .(not in seq logo color_dict).
    alphabet = 'ACDEFGHIKLMNPQRSTVWY.'
    alphabet_ascii = list(sorted([ord(x) for x in alphabet]))
    seq_ctrs = seq_ctrs[:, alphabet_ascii]

    # %% Normalise rows to have unit sum
    # row_sums = seq_ctrs.sum(axis=1)
    # seq_ctrs_norm = seq_ctrs / row_sums[:, np.newaxis]

    # %% Remove rows with all zeros (causes error with logomaker)
    seq_ctrs = seq_ctrs[~np.all(seq_ctrs == 0, axis=1)]

    # %% Pandas column headers
    column_headers = [chr(aa_ascii) for aa_ascii in alphabet_ascii]

    # Create pandas dataframe
    df_ctrs = pd.DataFrame(data=seq_ctrs, columns=column_headers)

    # %% Keep a copy with the gaps still in
    # (will return to show as list of sequences)
    df_ctrs_with_gaps = df_ctrs.copy()  # Return copy with the gaps still in.

    # Filter out positions which are a gap in >=95% of cases.
    # (We wouldn't see the 5% residues in the sequence logo, so remove
    # them to avoid an empty gap showing in the logo).
    #df_ctrs['_row_total'] = df_ctrs.sum(axis='columns')
    df_ctrs['_gap_pct'] = df_ctrs.apply(
        lambda row: row['.'] / row.sum(), axis='columns')
    is_not_mostly_gap = df_ctrs['_gap_pct'] < 0.95
    df_ctrs = df_ctrs[is_not_mostly_gap]
    del df_ctrs['_gap_pct']
    df_ctrs = df_ctrs.reset_index()
    df_ctrs = df_ctrs.drop(labels='index', axis='columns')

    # Remove gap '.' chars, so that sequence logo
    # sequence logo shows white gaps instead of black
    # (logo will re-normalise automatically)
    del df_ctrs['.']

    # %% Generate logo
    # %%time
    logo = logomaker.Logo(
        df_ctrs,
        fade_probabilities=True,
        stack_order='small_on_top',
        color_scheme='hydrophobicity',
        show_spines=True  # ,
        #font_name='DejaVu Sans Mono'
    )

    # Turn off the axes
    # TODO: Remove to switch back on
    logo.ax.axis('off')

    # Add title
    if title != '':
        logo.fig.suptitle(title)

    # Encode to base64 if requested
    if encode_base64:
        byts = io.BytesIO()
        logo.fig.set_size_inches(8, 1.5)
        logo.fig.tight_layout()
        logo.fig.savefig(
            byts,
            format=image_format,
            transparent=True,
            dpi=150,
        )
        byts.seek(0)
        logo_ret = base64.b64encode(byts.getvalue())\
            .decode('utf-8').replace('\n', '')
        logo_ret = f'data:image/{image_format.lower()};base64,{logo_ret}'
    else:
        logo_ret = logo

    # Return
    return logo_ret, df_ctrs_with_gaps


def get_region_positions(df_flat, record_id_column, redundancy_column):
    """
    Get the residue position where each region starts.
    Used for text annotation on the sequence logo.
    """

    # init
    fields_to_exclude = set([record_id_column, redundancy_column])
    column_list = [x for x in df_flat.columns if x not in fields_to_exclude]
    last_region_pos = 0
    pos = 0
    last_region = ''
    region_start_positions = {}

    # Loop and get positions
    for x in column_list:

        pos += 1
        region = x.split('.')[0]

        # First item
        if pos == 0:
            region_start_positions[region] = pos - 1
            last_region = region
            continue

        # Remaining items
        if region != last_region:
            region_start_positions[region] = pos - 1

        last_region = region

    # Return
    return region_start_positions


def add_region_annotations(ax, region_positions):
    """Add region annotations to the logo plot"""

    for region, pos in region_positions.items():

        # Regions
        ax.text(
            x=pos - 0.5,
            y=1.05,
            s=f'{region.upper()} >',
            fontfamily='sans-serif',
            fontsize='small'
        )

        # Line
        ax.plot(
            [pos - 0.5, pos - 0.5],
            [0., 1.]
        )
