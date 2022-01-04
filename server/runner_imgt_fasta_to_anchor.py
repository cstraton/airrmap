# Take an IMGT reference fasta file,
# translate the nucleotide sequences
# and produce a csv file that can be used
# for the anchor sequences.

# For IMGT reference sequences, see:
# http://www.imgt.org/vquest/refseqh.html

# Imports
import json
import argparse

import airrmap.preprocessing.imgt as imgt


def run(args):
    # Parse fasta
    df_imgt = imgt.parse_fasta_file(args.in_file)

    # Ensure JSON double quotes instead of single quotes
    # when writing annotated a.a. sequence (dict) to csv
    df_imgt['aa_annotated'] = df_imgt['aa_annotated'].apply(json.dumps)

    # Add region fields from gapped sequence (cdr1_aa, cdr2_aa etc.)
    regions = imgt.get_region_positions(
        chain='',  # e.g. just 'cdr1', not 'cdrh1'
        as_dict=False,
        is_rearranged=True,  # required if for_gapped_seq is True, can run to fw4
        for_gapped_seq=True
    )
    for region in regions:
        region_name, from_char_index, to_char_index = region
        to_char_index += 1  # orig is inclusive, make exclusive for slice
        f1_gapped = f'{region_name}_aa_gapped'
        f2 = f'{region_name}_aa'
        df_imgt[f1_gapped] = df_imgt['aa_gapped'].apply(
            lambda x: x[from_char_index:to_char_index]
        )
        df_imgt[f2] = df_imgt[f1_gapped].apply(
            lambda x: x.replace('.', '')
        )

    # Save
    df_imgt.to_csv(args.out_file)
    print(f'{len(df_imgt)} record(s) written to {args.out_file}')


def main():
    parser = argparse.ArgumentParser(
        description="Translate IMGT fasta gapped nucleotide reference sequences (V region) to anchors csv file."
    )
    parser.add_argument(
        '-i', '--in',
        help='The IMGT reference file, e.g. IGHV.fasta',
        dest='in_file',
        type=str,
        required=True
    )
    parser.add_argument(
        '-o', '--out',
        help='The output file, e.g. anchors.csv',
        dest='out_file',
        type=str,
        required=True
    )
    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


# Main
if __name__ == '__main__':
    main()
