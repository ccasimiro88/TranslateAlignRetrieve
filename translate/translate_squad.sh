#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/env
source ${ENV_DIR}/bin/activate

SQUAD_FILE=$1
LANG_SRC=$2
LANG_TGT=$3
OUTPUT_DIR=$4
RETRIEVE_FROM_ALIGNMENT=$5
VERSION2=$6

if [[ -z "$5" ]]; then
    python src/translate_squad.py \
           -squad_file  ${SQUAD_FILE} \
           -lang_source ${LANG_SRC} \
           -lang_target ${LANG_TGT} \
           -output_dir ${OUTPUT_DIR} \
           ${VERSION2}
else
    python src/translate_squad.py \
           -squad_file  ${SQUAD_FILE} \
           -lang_source ${LANG_SRC} \
           -lang_target ${LANG_TGT} \
           -output_dir ${OUTPUT_DIR} \
           ${RETRIEVE_FROM_ALIGNMENT} \
           ${VERSION2}
fi
