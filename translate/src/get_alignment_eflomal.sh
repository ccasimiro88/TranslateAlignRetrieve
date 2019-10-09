#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/../env
source ${ENV_DIR}/bin/activate

# Train alignment model with eflomal: '/home/casimiro/projects/hutoma/qa_bert_fine-tuning/env'
# The text should be tokenized before computing alignment. However, for some uses like the SQUAD
# context alignment, we want the compute the it with the raw text
FILE_SRC=$1
LANG_SRC=$2
FILE_TGT=$3
LANG_TGT=$4
ALIGNMENT_TYPE=$5
OUTPUT_FILE=$6

# Compute forward and reverse alignment models
TOOLS_DIR=${SCRIPT_DIR}/../tools
EFLOMAL_DIR=${TOOLS_DIR}/eflomal
FASTALIGN_DIR=${TOOLS_DIR}/fast_align
MOSES_DIR=${TOOLS_DIR}/mosesdecoder

export LC_ALL=en_US.UTF8
tokenize(){
    INPUT_FILE=$1
    INPUT_LANG=$2
    cat ${INPUT_FILE} | \
      perl ${MOSES_DIR}/scripts/tokenizer/tokenizer.perl -l ${INPUT_LANG} < ${INPUT_FILE}
}

echo 'Compute alignments...'
FWD_ALIGN=$(mktemp)
REV_ALIGN=$(mktemp)
SYM_ALIGN=$(mktemp)

python ${EFLOMAL_DIR}/align.py -s ${FILE_SRC} -t ${FILE_TGT} --model 3 -f ${FWD_ALIGN} -r ${REV_ALIGN} -v --overwrite

echo "Symmetrize alignments..."
${FASTALIGN_DIR}/build/atools -c grow-diag-final-and -i ${FWD_ALIGN} -j ${REV_ALIGN} > ${SYM_ALIGN}

if [[ "$ALIGNMENT_TYPE" == "forward" ]]; then
  cp ${FWD_ALIGN} ${OUTPUT_FILE}
elif [[ "$ALIGNMENT_TYPE" == "reverse" ]]; then
  cp ${REV_ALIGN} ${OUTPUT_FILE}
elif [[ "$ALIGNMENT_TYPE" == "symmetric" ]]; then
  cp ${SYM_ALIGN} ${OUTPUT_FILE}
fi

rm ${FWD_ALIGN} ${REV_ALIGN} ${SYM_ALIGN}