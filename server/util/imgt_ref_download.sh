#!/usr/bin/env bash

# Downloads the IMGT VDJ heavy-chain reference nucleotide gapped sequences
# to ./imgt-ref subfolder


# init
set -o errexit
set -o nounset
set -o pipefail


# -N: don't retrieve files unless newer than local
# -P: save to subfolder
main() {
  
    subfolder=./imgt-ref
    wget http://www.imgt.org/download/V-QUEST/IMGT_V-QUEST_reference_directory/Homo_sapiens/IG/IGHD.fasta -N -P ${subfolder}
    wget http://www.imgt.org/download/V-QUEST/IMGT_V-QUEST_reference_directory/Homo_sapiens/IG/IGHJ.fasta -N -P ${subfolder}
    wget http://www.imgt.org/download/V-QUEST/IMGT_V-QUEST_reference_directory/Homo_sapiens/IG/IGHV.fasta -N -P ${subfolder}
}

main "${@}" # pass args to main

