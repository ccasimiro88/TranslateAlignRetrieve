#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/../../env
source ${ENV_DIR}/bin/activate

# Train the alignment model with eflomal and generated priors
# The text should be tokenised before computing the alignment.
FILE_SRC=$1
LANG_SRC=$2
FILE_TGT=$3
LANG_TGT=$4

if [[ $# -eq 0 ]]
  then
  FILE_SRC=${SCRIPT_DIR}/../nmt/data/en2es/preprocess/train.tok.en
  LANG_SRC=en
  FILE_TGT=${SCRIPT_DIR}/../nmt/data/en2es/preprocess/train.tok.es
  LANG_TGT=es
fi

# Compute forward and reverse alignment models
TOOLS_DIR=${SCRIPT_DIR}/../../tools
EFLOMAL_DIR=${TOOLS_DIR}/eflomal
FASTALIGN_DIR=${TOOLS_DIR}/fast_align
MOSES_DIR=${TOOLS_DIR}/mosesdecoder

DATA_DIR=${SCRIPT_DIR}/data
mkdir -p ${DATA_DIR}

export LC_ALL=en_US.UTF8

echo 'Train the alignment model...'
FWD_ALIGN=$(mktemp)
REV_ALIGN=$(mktemp)
SYM_ALIGN=$(mktemp)

# Tokenize and convert into FastAlign format
FILE_SRC_TGT_TOK=$(mktemp)
paste -d '|' ${FILE_SRC} ${FILE_TGT} \
    | sed 's/|/ ||| /g' \
    > ${FILE_SRC_TGT_TOK}

python ${EFLOMAL_DIR}/align.py \
    -i ${FILE_SRC_TGT_TOK} \
    --model 3 \
    -f ${FWD_ALIGN} \
    -r ${REV_ALIGN} \
    -v --overwrite

cp ${FWD_ALIGN} ${DATA_DIR}/align.fwd."${LANG_SRC}"-"${LANG_TGT}"
cp ${REV_ALIGN} ${DATA_DIR}/align.rev."${LANG_SRC}"-"${LANG_TGT}"

python ${EFLOMAL_DIR}/makepriors.py \
    -i ${FILE_SRC_TGT_TOK} \
    -f ${DATA_DIR}/align.fwd."${LANG_SRC}"-"${LANG_TGT}" \
    -r ${DATA_DIR}/align.rev."${LANG_SRC}"-"${LANG_TGT}" \
    --priors ${DATA_DIR}/align.priors."${LANG_SRC}"-"${LANG_TGT}" \


rm ${FWD_ALIGN} ${REV_ALIGN} ${SYM_ALIGN} ${FILE_SRC_TGT_TOK}













