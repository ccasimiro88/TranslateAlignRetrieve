# This script create the Train/Dev/Test datasets.
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/../../env/bin
source $ENV_DIR/activate

SRC_FILE=$1
SRC_LANG=$2
TGT_FILE=$3
TGT_LANG=$4

# Create datasets dir
SRC_TO_TGT=$SRC_LANG'2'$TGT_LANG
DATASETS_DIR=${SCRIPT_DIR}/data/${SRC_TO_TGT}/datasets
mkdir -p ${DATASETS_DIR}
python ${SCRIPT_DIR}/src/create_datasets.py
    --source_file ${SRC_FILE} \
    --source_lang ${SRC_LANG} \
    --target_file ${TGT_FILE} \
    --target_lang ${TGT_LANG} \
    --output_dir ${DATASETS_DIR} \
    --test_size 1000 \
    --valid_size 5000


