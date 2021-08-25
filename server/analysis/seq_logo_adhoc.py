# Generate a sequence logo using LogoMaker
# (Used for figures)

# %% Imports
import numpy as np
import airrmap.application.seqlogo as seqlogo

# %% # CONFIG HERE
gapped_seqs = np.array([
                'GFTFSSYWIKQDGSEK'
            ])

# %% Generate the logo
logo, df_ctrs_with_gaps = seqlogo.get_logo(
    gapped_seqs
)
