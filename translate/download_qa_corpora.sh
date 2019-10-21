#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SQUAD_DIR=${SCRIPT_DIR}/corpora/squad_v2
mkdir -p ${SQUAD_DIR}

# Donwload SQUAD
SQUAD_TRAIN="https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
SQUAD_DEV="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
wget -nc -P ${SQUAD_DIR} ${SQUAD_TRAIN}
wget -nc -P ${SQUAD_DIR} ${SQUAD_DEV}