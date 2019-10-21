#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Create virtualenv
ENV_DIR=$SCRIPT_DIR/env
virtualenv -p python3.6 $ENV_DIR
source $ENV_DIR/bin/activate

# Create structure directories
mkdir -p $SCRIPT_DIR/data/alignment/eflomal
mkdir -p $SCRIPT_DIR/data/translate_squad
mkdir -p $SCRIPT_DIR/data/fastText
mkdir -p $SCRIPT_DIR/data/nmt/en2es

# Install python dependencies
pip install -r $SCRIPT_DIR/requirements.txt

# Install non-python libraries
TOOLS_DIR=$SCRIPT_DIR/tools
mkdir -p $TOOLS_DIR

# Eflomal: 'https://github.com/robertostling/eflomal'
EFLOMAL_DIR=$TOOLS_DIR/eflomal
git clone https://github.com/robertostling/eflomal.git $TOOLS_DIR/eflomal
cd $EFLOMAL_DIR
make
make install
pip install .
cd $SCRIPT_DIR

# FastAlign: 'https://github.com/clab/fast_align.git'
apt-get install libgoogle-perftools-dev libsparsehash-dev

FASTALIGN_DIR=$TOOLS_DIR/fast_align
git clone https://github.com/clab/fast_align.git $FASTALIGN_DIR
cd $FASTALIGN_DIR
mkdir build
cd build
cmake ..
make
cd $SCRIPT_DIR

# Moses
MOSES_DIR=$TOOLS_DIR/mosesdecoder
git clone https://github.com/moses-smt/mosesdecoder.git $MOSES_DIR

# OpenNMT
ONMT_DIR=$TOOLS_DIR/OpenNMT-py
git clone https://github.com/OpenNMT/OpenNMT-py.git $ONMT_DIR

# Download FastText model
FASTTEXT_DIR=$SCRIPT_DIR/data/fastText
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P $FASTTEXT_DIR

