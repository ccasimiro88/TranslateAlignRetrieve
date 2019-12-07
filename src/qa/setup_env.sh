#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Create virtualenv
ENV_DIR=${SCRIPT_DIR}/env
virtualenv -p python3.6 ${ENV_DIR}
source ${ENV_DIR}/bin/activate

# Install python dependencies
pip install -r ${SCRIPT_DIR}/requirements.txt

# Install non-python libraries
TOOLS_DIR=${SCRIPT_DIR}/tools
mkdir -p ${TOOLS_DIR}

# Transformers
TRANSFORMERS_DIR=${TOOLS_DIR}/transformers
git clone https://github.com/huggingface/transformers.git ${TRANSFORMERS_DIR}

# MLQA
MLQA_DIR=${TOOLS_DIR}/MLQA
git clone https://github.com/facebookresearch/MLQA.git ${TOOLS_DIR}/MLQA
