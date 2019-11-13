#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=${SCRIPT_DIR}/env

virtualenv -p python3.6 ${ENV_DIR}
source ${ENV_DIR}/bin/activate

pip install -r ${SCRIPT_DIR}/requirements.txt

TRANSFORMERS_DIR=${SCRIPT_DIR}/transformers
TOOLS_DIR=${SCRIPT_DIR}/tools
mkdir -p ${TRANSFORMERS_DIR}
mkdir -p ${TOOLS_DIR}

git clone https://github.com/huggingface/transformers.git ${TOOLS_DIR}/transformers
pip install ${TOOLS_DIR}/transformers

git clone https://github.com/facebookresearch/MLQA.git ${TOOLS_DIR}/MLQA
