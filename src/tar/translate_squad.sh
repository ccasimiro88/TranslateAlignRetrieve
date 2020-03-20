#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/env
source ${ENV_DIR}/bin/activate

SQUAD_FILE=$1
OUTPUT_DIR=$2
RETRIEVE_FROM_ALIGNMENT=$3

LANG_SRC=en
LANG_TGT=es

TRANSLATE_RETRIEVE_DIR=${SCRIPT_DIR}/src/retrieve
if [[ -z "$3" ]]; then
    python ${TRANSLATE_RETRIEVE_DIR}/translate_retrieve_squad.py \
           -squad_file  ${SQUAD_FILE} \
           -lang_source ${LANG_SRC} \
           -lang_target ${LANG_TGT} \
           -output_dir ${OUTPUT_DIR}
else
    python ${TRANSLATE_RETRIEVE_DIR}/translate_retrieve_squad.py \
           -squad_file  ${SQUAD_FILE} \
           -lang_source ${LANG_SRC} \
           -lang_target ${LANG_TGT} \
           -output_dir ${OUTPUT_DIR} \
           ${RETRIEVE_FROM_ALIGNMENT}
fi
