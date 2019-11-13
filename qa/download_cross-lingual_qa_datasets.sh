# This script download the available cross-lingual Question Answering data-sets:
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Download MLQA: https://github.com/facebookresearch/MLQA
CORPORA_DIR=${SCRIPT_DIR}/corpora
mkdir -p ${CORPORA_DIR}
wget -nc -P ${CORPORA_DIR} "https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip"
unzip -o ${CORPORA_DIR}/MLQA_V1.zip -d ${CORPORA_DIR}
rm ${CORPORA_DIR}/MLQA_V1.zip