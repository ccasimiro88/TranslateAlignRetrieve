# The Translate-Align-Retrieve (TAR) method for synthetic QA corpora generation

This repository contains the implementation of the TAR method designed and implemented
for the automatic translation of the Stanford Question Answering Dataset (SQuAD) into Spanish.
There are three folder containing:
 
- `src/tar`: The code of the TAR method
- `SQuAD-es-v1.1`: The resulting Spanish translations of the SQuAD v1.1 training dataset
- `SQuAD-es-v2.0`: The resulting Spanish translations of the SQuAD v2.0 training dataset
- `src/qa`: The code to train a QA system based on the pre-trained Multilingual-BERT model and evaluate it.
The code is based on the HuggingFace library: 

## TAR description

The TAR method is composed of three independent components that processca QA dataset composed
of (context, question, answer) tuple and return its translation:

 1. A machine translation model from the source to the target language
 2. An word alignment model for source and target sentences
 3. An answer retrieval component to translated 
 
Currently, the machine translation component is a English to Spanish translator applied to the 
SQuAD dataset to generate the corresponding SQuAD-es datasets. However, it can be extended for the
generation of translation in other target languages and the processing of different QA datasets, 
Natural Questions (cite), for example.

For a detailed description of the TAR method refer to the paper 
**Automatic Spanish Translation of SQuAD Dataset for Multi-lingual Question Answering**
listed in the references. 

 
### Installation
All the python dependencies and other non-python libraries can be automatically installed
by running the script `src/tar/setup_env.sh`

### NMT training
First, to train a neural machine translation system from English to Spanish based on the Transformer model, 
the following steps are performed by executing the scripts under the directory `src/tar/src/nmt/`: 

1. Download the en-es parallel corpora and create train/valid/test dataset :

    `download_en-es_corpora.sh && create_datasets.sh`
    
2. Preprocess the train/valid datasets to with Byte-Pair Encoding (BPE): 

    `preprocess.sh`
    
3. Train the NMT model with shared source/target vocabulary and embeddings:

    `train_shared.sh`
  
4. Average last three model checkpoints

   `average_models.sh`
   
5. Evaluate the final average NMT model on the test set with BLEU score

    `evaluate.sh`

### Alignment
Second, to train the word alignment model on the previously generated tokenised train sets 
`train.tok.en` and `train.tok.es` execute the following script under 
the directory `src/tar/src/alignment`:

- `train_alignment_with_priors.sh`

### Translate and retrieve
Eventually, to generate the Spanish translation of the SQUAD datasets, both version v1.1 and v2.0,
run one of the following commands under the directory `src/tar/src/retrieve`

1. Download the SQuAD dataset, both version v1.1 and v2.0

    `download_squad.sh`

1A. Generate the full dataset translation, called SQuAD-es:

    `translate_squad.sh <squad_file> -answers_from_alignment`
    
1B. Optionally, generate the small dataset translation, called SQuAD-es-small:

    `translate_squad.sh <squad_file>`

The option 1A. refers to a big, with 100% of the original SQuAD data, but noisy dataset with all the 
possible retrieved (context, question, answer). The option 1B refers to the small, with about 53% of the original SQuAD
data, but more accurate translation. 


## Question answering
This section shows how to train a question answering (QA) system with the SQuAD-es training dataset by fine-tuning 
a Multilingual-BERT (mBERT) model. The evaluation is performed on the recently proposed 
MLQA (cite) and XQuAD (cite) benchmarks for cross-lingual QA evaluation.

### Installation
All the python dependencies and other non-python libraries can be automatically installed
by running the script `src/qa/setup_env.sh`

### Train
To train a QA system, run the following script under the directory `src/qa`:

    `train_m-bert.sh <squad_file> <squad_version>`

The first argument the SQuAD training file and the second argument is a string with the version, namely as v1 or v2.
The resulting model will be saved in the default directory `data/training/m-bert_<squad_file>`

### Evaluate

To evaluate the trained QA systems, run the following script under the directory `src/qa`:

1. Download the MLQA and XQuAD evaluation ben:

    `download_mlqa_xquad.sh`
    
2. Evaluate on the MLQA or XQuAD:

    `evaluate_m-bert.sh <qa_model_dir> <context_lang> <question_lang> <test_set>`
    
    *qa_model_dir* is the directory containing the trained qa model
    
    *context_lang* and *question_lang* are the 
    question and context language expressed in the ISO 639-1 notation (*en*, *es*, etc...)
     
    *test_set* is the name of the evaluation set, namely *mlqa* of *xquad*

    The F1 and Exact Match score are computed with the MLQA evaluation script that is a 
    multilingual generalization of the commonly-used SQuAD evaluation script.
    The resulting scores are written in a file under the directory `data/evaluate`