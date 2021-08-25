# Use anarci to number a single sequence
# and show the regions, original sequence index,
# and gapped sequence index for each residue.
# Used for the antibody structure figure, in order
# to map the residues from PDB / in Chimera to regions / IMGT-numbered residues.
# Requires ANARCI to be installed.
# (see https://github.com/oxpig/ANARCI)

# USE:
# 1. Set the properties in Init (only first sequence will be used)
# 2. Run all

# TO VERIFY IMGT NUMBERING:
# http://opig.stats.ox.ac.uk/webapps/newsabdab/sabpred/anarci/

# EXAMPLE INPUT:
# seq_list = [
#   ('ID1', 'EVQLVESGAEVKKPGSSVKVSCKASGGPFRSYAISWVRQAPGQGPEWMGGIIPIFGTTKYAPKFQGRVTITADDFAGTVYMELSSLRSEDTAMYYCAKHMGYQVRETMDVWGKGTTVTVSSASTKGPSVFPLAPGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEP')
# ]

# EXAMPLE OUTPUT:
#   orig_str_index    gapped_index  imgt_name    residue    region_name    conserved_name
# ----------------  --------------  -----------  ---------  -------------  ----------------
#                1               0  1            E          fwh1
#                2               1  2            V          fwh1
#                3               2  3            Q          fwh1
# ...
#               20              20  21           V          fwh1
#               21              21  22           S          fwh1
#               22              22  23           C          fwh1           1st-CYS
# ...
#              104             143  112B         -          cdrh3
#              104             144  112A         V          cdrh3
#              105             145  112          R          cdrh3
#              106             146  113          E          cdrh3
# ...

# %% Imports
from anarci.anarci import run_anarci as run_anarci
import airrmap.preprocessing.imgt as imgt
from tabulate import tabulate  # type: ignore

# %% Init
seq_list = [
    ('5whj.H', 'EVQLLESGGGLVQPGGSLRLSCAASGFTFSEYAMGWVRQAPGKGLEWVSSIGSSGGQTKYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARLAIGDSYWGQGTMVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSC')
    #('5whj.L', 'QSALTQPASVSGSPGQSITISCTGTGSDVGSYNLVSWYQQHPGKAPKLMIYGDSQRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCASYAGSGIYVFGTGTKVTVLGQPKANPTVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADGSPVKAGVETTKPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECS')
]
scheme = 'imgt'  # 'aho'
chain = 'h'
allow = set(chain.upper())  # set(["H","K","L","A","B","G","D"])
allowed_species = ['human']
assign_germline = False
ncpu = 4
gap_char = '-'
orig_seq_index_start = 1

# %% ANARCI number
orig_seq_list, numbered, align_details, hit_table = run_anarci(
    seq_list,
    scheme=scheme,
    output=False,
    allow=allow,  # set(["H","K","L","A","B","G","D"])
    # ['human', 'mouse','rat','rabbit','rhesus','pig','alpaca']
    allowed_species=allowed_species,
    assign_germline=assign_germline,
    ncpu=ncpu
)

# %% Get first numbered seq
numbered_seq = numbered[0][0][0]

# %% Convert to OAS format
# [((112, 'A'), 'V'), (112, ' '), 'R']
# --> {'all': {'112A': 'V', '112': R...}}
imgt_seq = dict(
    all={f'{item[0][0]}{item[0][1].strip()}': item[1] for item in numbered_seq}
)

# %% Get mappings from IMGT name to index
imgt_residue_names = imgt.get_imgt_residue_names(as_dict=True)

# %% Get gapped sequence
seq_gapped = imgt.imgt_to_gapped(
    imgt_seq,
    imgt_residue_names=imgt_residue_names,  # type: ignore
    gap_char=gap_char
)

# %% Get region positions for gapped sequence
region_positions = imgt.get_region_positions(
    chain=chain,
    as_dict=False,
    is_rearranged=True,
    for_gapped_seq=True
)


# %% IMGT - conserved residues (useful pointers)
# REF: http://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
# REF: http://www.imgt.org/3Dstructure-DB/doc/IMGTCollier-de-Perles.shtml
conserved_residues = {
    '23': '1st-CYS',
    '41': 'CONSERVED-TRP',
    '89': 'Hydrophobic 89',
    '118': 'J-PHE or J-TRP',
    '119': 'Glycine 119'
}


# %% Build and print tall list of residues
# with regions and original vs gapped residue indexes.
original_index = orig_seq_index_start
gapped_index = 0
region_index = -1
current_region_from = -1
current_region_to = -1
record_list = []
gapped_index_to_imgt_name = {v: k for k,
                             v in imgt_residue_names.items()}  # type: ignore


for char in seq_gapped:

    # Get the current CDR or framework region
    while not (gapped_index >= current_region_from and gapped_index <= current_region_to):
        region_index += 1
        current_region_name = region_positions[region_index][0]
        current_region_from = region_positions[region_index][1]
        current_region_to = region_positions[region_index][2]

    # Get IMGT residue name from Gapped Seq
    imgt_residue_name = gapped_index_to_imgt_name[gapped_index]

    # Build the record
    record = dict(
        orig_str_index=original_index,
        gapped_index=gapped_index,
        imgt_name=imgt_residue_name,
        residue=char,
        region_name=current_region_name,
        conserved_name=conserved_residues[imgt_residue_name] if str(
            imgt_residue_name) in conserved_residues else ''
    )
    record_list.append(record)

    # Increment indexes
    gapped_index += 1
    if char != gap_char:
        original_index += 1


# %% Show the regions
#result = []
# for region in region_positions:
#    region_name, region_from, region_to = region
#    result.append(
#        (region_name, region_from, region_to, seq_gapped[region_from:region_to+1])
#    )


# %% Show
print(tabulate(record_list, headers='keys'))
# %%
