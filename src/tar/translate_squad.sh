#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/env
source ${ENV_DIR}/bin/activate

SQUAD_FILE=$1
OUTPUT_DIR=$2
LANG_SRC=$3
LANG_TGT=$4
RETRIEVE_FROM_ALIGNMENT=$5

if [[ -z "$3" ]] && [[ -z "$4" ]]
  then
  LANG_SRC=en
  LANG_TGT=es
fi

TRANSLATE_RETRIEVE_DIR=${SCRIPT_DIR}/src/retrieve
if [[ -z "$5" ]]; then
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
