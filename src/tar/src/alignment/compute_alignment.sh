#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/../../env
source ${ENV_DIR}/bin/activate

# Compute the source-translation alignment with eflomal
# The text should be tokenized before computing alignment.
FILE_SRC=$1
FILE_TGT=$2
ALIGNMENT_TYPE=$3
OUTPUT_FILE=$4
PRIORS_DIR=$5

export LC_ALL=en_US.UTF8

# Compute forward and reverse alignment models
TOOLS_DIR=${SCRIPT_DIR}/../../tools
EFLOMAL_DIR=${TOOLS_DIR}/eflomal
FASTALIGN_DIR=${TOOLS_DIR}/fast_align

echo 'Compute alignments...'
FWD_ALIGN=$(mktemp)
REV_ALIGN=$(mktemp)
SYM_ALIGN=$(mktemp)

if [[ -n $PRIORS_DIR ]]; then
  python ${EFLOMAL_DIR}/align.py \
          -s ${FILE_SRC} \
          -t ${FILE_TGT} \
          --priors ${PRIORS_DIR}/align.priors* \
          --model 3 \
          -f ${FWD_ALIGN} \
          -r ${REV_ALIGN} \
          -v --overwrite
else
    python ${EFLOMAL_DIR}/align.py \
          -s ${FILE_SRC} \
          -t ${FILE_TGT} \
          --model 3 \
          -f ${FWD_ALIGN} \
          -r ${REV_ALIGN} \
          -v --overwrite
fi

echo "Symmetrize alignments..."
${FASTALIGN_DIR}/build/atools \
    -c grow-diag-final-and \
    -i ${FWD_ALIGN} \
    -j ${REV_ALIGN} \
    > ${SYM_ALIGN}

if [[ "$ALIGNMENT_TYPE" == "forward" ]]; then
  cp ${FWD_ALIGN} ${OUTPUT_FILE}
elif [[ "$ALIGNMENT_TYPE" == "reverse" ]]; then
  cp ${REV_ALIGN} ${OUTPUT_FILE}
elif [[ "$ALIGNMENT_TYPE" == "symmetric" ]]; then
  cp ${SYM_ALIGN} ${OUTPUT_FILE}
fi

rm ${FWD_ALIGN} ${REV_ALIGN} ${SYM_ALIGN}
echo "alignment file wrote to: $(realpath $OUTPUT_FILE)"