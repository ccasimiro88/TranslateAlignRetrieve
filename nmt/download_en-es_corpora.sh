#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR=$SCRIPT_DIR/env/bin
source $ENV_DIR/activate

CORPORA_DIR=${SCRIPT_DIR}/corpora
mkdir -p ${CORPORA_DIR}

EN_ES_DIR=${CORPORA_DIR}/en-es

echo "Download en-es corpora..."
# WikiMatrix from LASER repo: https://github.com/facebookresearch/LASER/tree/master/tasks/WikiMatrix
echo "WikiMatrix from LASER repository..."
wget -nc -P ${EN_ES_DIR}/laser_wikimatrix https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.en-es.tsv.gz

LASER_DIR=${SCRIPT_DIR}/tools/laser
python ${LASER_DIR}/tasks/WikiMatrix/extract.py \
       --tsv ${EN_ES_DIR}/laser_wikimatrix/WikiMatrix.en-es.tsv.gz \
       --bitext ${EN_ES_DIR}/laser_wikimatrix/WikiMatrix.en-es.txt \
       --src-lang en --trg-lang es \
       --threshold 1.04


# Wikipedia from OPUS: http://opus.nlpl.eu/Wikipedia-v1.0.php
echo "Wikipedia from OPUS..."
wget -nc -P ${EN_ES_DIR}/opus_wikipedia_v1 https://object.pouta.csc.fi/OPUS-Wikipedia/v1.0/moses/en-es.txt.zip
unzip -o ${EN_ES_DIR}/opus_wikipedia_v1/en-es.txt.zip -d ${EN_ES_DIR}/opus_wikipedia_v1

# Ted from OPUS: http://opus.nlpl.eu/TED2013-v1.1.php
echo "TED subtitles from OPUS..."
wget -nc -P ${EN_ES_DIR}/opus_ted2013 https://object.pouta.csc.fi/OPUS-TED2013/v1.1/moses/en-es.txt.zip
unzip ${EN_ES_DIR}/opus_ted2013/en-es.txt.zip -d ${EN_ES_DIR}/opus_ted2013

# News-Commentary from OPUS: v14
echo "News-Commentary from OPUS..."
wget -nc -P ${EN_ES_DIR}/opus_news-commentary_v14 https://object.pouta.csc.fi/OPUS-News-Commentary/v14/moses/en-es.txt.zip
unzip -o ${EN_ES_DIR}/opus_news-commentary_v14/en-es.txt.zip -d ${EN_ES_DIR}/opus_news-commentary_v14

# Tatoeba from OPUS: http://opus.nlpl.eu/Tatoeba-v20190709.php
echo "Tatoeba from OPUS..."
wget -nc -P ${EN_ES_DIR}/opus_tatoeba_v20190709 https://object.pouta.csc.fi/OPUS-Tatoeba/v20190709/moses/en-es.txt.zip
unzip -o ${EN_ES_DIR}/opus_tatoeba_v20190709/en-es.txt.zip -d ${EN_ES_DIR}/opus_tatoeba_v20190709

# OpenSubTitles from OPUS: http://opus.nlpl.eu/OpenSubtitles-v2018.php
echo "OpenSubTitles from OPUS"
wget -nc -P ${EN_ES_DIR}/opus_opensubtitles_v2018 https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-es.txt.zip
unzip -o ${EN_ES_DIR}/opus_opensubtitles_v2018/en-es.txt.zip -d ${EN_ES_DIR}/opus_opensubtitles_v2018

# Select just the first MAX_SIZE lines
MAX_SIZE=2500000
head -n $MAX_SIZE ${EN_ES_DIR}/opus_opensubtitles_v2018/OpenSubtitles.en-es.en > \
  ${EN_ES_DIR}/opus_opensubtitles_v2018/opensubtitles.en-es.en
head -n $MAX_SIZE ${EN_ES_DIR}/opus_opensubtitles_v2018/OpenSubtitles.en-es.es > \
  ${EN_ES_DIR}/opus_opensubtitles_v2018/opensubtitles.en-es.es
rm ${EN_ES_DIR}/opus_opensubtitles_v2018/OpenSubtitles.en-es.e*

# Merge all te datasets in one corpora
echo 'Merge all the corpora...'
cat $(find ${EN_ES_DIR}  -name "*.en") > ${EN_ES_DIR}/corpora.en
cat $(find ${EN_ES_DIR}  -name "*.es") > ${EN_ES_DIR}/corpora.es

echo 'Final corpora size..'
echo $(cat ${EN_ES_DIR}/corpora.es | wc -l) ' lines'