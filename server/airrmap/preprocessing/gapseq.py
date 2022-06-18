"""Gapped sequence functions"""

# tests > test_gapseq.py


def get_gapped_seq(seq: str, fixed_length: int, gap_char: str = '.', left_bias=True) -> str:
    """
    Create a gapped sequence string.

    Splits 'seq' into two halves and inserts gap characters
    so that the length of the returned gapped string equals
    'fixed_length'.

    Parameters
    ----------
    seq : str
        The original string sequence.

    fixed_length : int
        The ouput fixed size of the string.

    gap_char : str, optional
        The gap character to use, by default '.'

    left_bias: bool, optional
        If 'seq' length is not even, then
        True to keep remaining residue on the left side,
        otherwise pass False for right.
        By default, True.

    Returns
    -------
    str
        The gapped string.

    Raises
    ------
    ValueError
        If fixed_length is less than the sequence length.
    """

    # Example gap introduction for CDR-IMGT (27-38)
    # REF: http://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
    # Length 12 	27	28	29	30	31	32	33	34	35	36	37	38
    # Length 11 	27	28	29	30	31	32	-	34	35	36	37	38
    # Length 10     27	28	29	30	31	-	-	34	35	36	37	38
    # Length 09 	27	28	29	30	31	-	-	-	35	36	37	38
    # etc.

    # Init
    seq_len = len(seq)

    # Compute the gap length
    gap_len = fixed_length - seq_len
    if (gap_len < 0):
        raise ValueError(
            f'fixed_len value ({fixed_length}) is too small for sequence length ({seq_len}: {seq}). Consider increasing envconfig.application.seq_logos.fixed_length property.'
        )

    # Split into two halves.
    # Handle non-even seq lengths.
    split_point = seq_len // 2
    if seq_len % 2 != 0:
        split_point += 1 if left_bias else 0  # +0 for right bias if non-even seq length

    # Return the gapped sequence
    # (left half + gap + right half)
    return seq[:split_point] + (gap_char * gap_len) + seq[split_point:]
