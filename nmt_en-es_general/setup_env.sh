#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Create virtualenv
ENV_DIR=$SCRIPT_DIR/env
virtualenv -p python3.6 $ENV_DIR
source $ENV_DIR/bin/activate

# Install python dependencies
pip install -r $SCRIPT_DIR/requirements.txt

# Install non-python libraries
TOOLS_DIR=$SCRIPT_DIR/tools
mkdir -p $TOOLS_DIR

# Moses
MOSES_DIR=$TOOLS_DIR/mosesdecoder
git clone https://github.com/moses-smt/mosesdecoder.git $MOSES_DIR

# OpenNMT
ONMT_DIR=$TOOLS_DIR/OpenNMT-py
git clone https://github.com/OpenNMT/OpenNMT-py.git $ONMT_DIR

# LASER
LASER_DIR=${TOOLS_DIR}/laser
git clone https://github.com/facebookresearch/LASER.git $LASER_DIR

# Download FastText model
DATA_DIR=${SCRIPT_DIR}/data
FASTTEXT_DIR=${DATA_DIR}/fastText
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P $FASTTEXT_DIR
