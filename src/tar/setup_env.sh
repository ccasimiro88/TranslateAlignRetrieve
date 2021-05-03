#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Create virtualenv
ENV_DIR=${SCRIPT_DIR}/env
virtualenv -p python3.6 ${ENV_DIR}
source ${ENV_DIR}/bin/activate

# Install python dependencies
pip install -r ${SCRIPT_DIR}/requirements.txt

# Install external repositories
TOOLS_DIR=${SCRIPT_DIR}/tools
mkdir -p ${TOOLS_DIR}

# Moses
MOSES_DIR=${TOOLS_DIR}/mosesdecoder
git clone https://github.com/moses-smt/mosesdecoder.git ${MOSES_DIR}

# LASER
LASER_DIR=${TOOLS_DIR}/laser
git clone https://github.com/facebookresearch/LASER.git ${LASER_DIR}

# Eflomal
EFLOMAL_DIR=${TOOLS_DIR}/eflomal
git clone https://github.com/robertostling/eflomal.git ${TOOLS_DIR}/eflomal
cd ${EFLOMAL_DIR}
mkdir ~/tmp_eflomal
export TMPDIR=~/tmp_eflomal
mkdir ${EFLOMAL_DIR}/bin
make
make install -e INSTALLDIR=${EFLOMAL_DIR}/bin
python setup.py install
cd ${SCRIPT_DIR}

# FastAlign
apt-get install libgoogle-perftools-dev libsparsehash-dev

FASTALIGN_DIR=${TOOLS_DIR}/fast_align
git clone https://github.com/clab/fast_align.git ${FASTALIGN_DIR}
cd ${FASTALIGN_DIR}
mkdir build
cd build
cmake ..
make
cd ${SCRIPT_DIR}

# OpenNMT
ONMT_DIR=${TOOLS_DIR}/OpenNMT-py
git clone https://github.com/OpenNMT/OpenNMT-py.git ${ONMT_DIR}