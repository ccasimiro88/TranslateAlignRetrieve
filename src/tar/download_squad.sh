#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SQUAD_DIR=${SCRIPT_DIR}/corpora/squad-en
mkdir -p ${SQUAD_DIR}

# Download SQUAD
# Version 2.0
SQUAD_TRAIN_v2="https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
SQUAD_DEV_v2="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
wget -nc -P ${SQUAD_DIR} ${SQUAD_TRAIN_v2}
wget -nc -P ${SQUAD_DIR} ${SQUAD_DEV_v2}

# Version 1.0
SQUAD_TRAIN_v1="https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
SQUAD_DEV_v1="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
wget -nc -P ${SQUAD_DIR} ${SQUAD_TRAIN_v1}
wget -nc -P ${SQUAD_DIR} ${SQUAD_DEV_v1}