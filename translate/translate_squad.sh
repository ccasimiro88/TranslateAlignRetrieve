#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/env
source ${ENV_DIR}/bin/activate

SQUAD_FILE=$1
LANG_SRC=$2
LANG_TGT=$3
OUTPUT_DIR=$4
BATCH_SIZE=$5

python src/translate_squad.py \
       -squad_file  ${SQUAD_FILE} \
       -lang_source ${LANG_SRC} \
       -lang_target ${LANG_TGT} \
       -output_dir ${OUTPUT_DIR} \
       -alignment_tokenized \
       ${BATCH_SIZE} \
       -retrieve_answers_from_alignment \
