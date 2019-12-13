# The Translate-Align-Retrieve (TAR) method for synthetic QA corpora generation

This repository contains the implementation of the TAR method designed and implemented
for the automatic translation of the Stanford Question Answering Dataset (SQuAD) into Spanish.
It is divided in four folders:
 
- `src/tar`: The code of the TAR method

- `SQuAD-es-v1.1`: The resulting Spanish translations of the SQuAD v1.1 training dataset

- `SQuAD-es-v2.0`: The resulting Spanish translations of the SQuAD v2.0 training dataset

- `src/qa`: The code used to train a QA system based on the pre-trained Multilingual-BERT model fine-tuned with the SQuAD-es datasets.
            Furthermore, is contains the code to evaluate the resulting system on two cross-lingual QA benchmarks.
            The code is based on the [HuggingFace's transforemer repository](https://github.com/huggingface/transformers): 


For a detailed description of this work, refer to the pre-print version of the paper 
**[Automatic Spanish Translation of SQuAD Dataset for Multi-lingual Question Answering](https://arxiv.org/abs/1912.05200v2)**.

## TAR description

The TAR method is composed of three independent components used to translate a QA dataset composed
of (context, question, answer) tuple:

 1. A machine translation model from the source to the target language
 2. An word alignment model for source and target sentences
 3. An answer retrieval component
 
Currently, the machine translation component is an English to Spanish translator applied to the 
SQuAD dataset to generate the corresponding SQuAD-es datasets, namely version 1.1 and 2.0. 
However, it can be extended for the other target languages and the different QA datasets, such as
[Natural Questions](https://github.com/google-research-datasets/natural-questions).

 
### Installation
All the python dependencies and other non-python libraries can be automatically installed
by running the script: `src/tar/setup_env.sh`

### NMT training
First, to train a neural machine translation system from English to Spanish based on the Transformer model, 
the following steps are performed by running the related scripts under the directory `src/tar/src/nmt/`: 

1. Download the en-es parallel corpora and split into train/valid/test dataset:

    `download_en-es_corpora.sh && create_datasets.sh`
    
2. Preprocess the train/valid datasets: 

    `preprocess.sh`
    
3. Train the NMT model with shared source/target vocabulary and embeddings:

    `train_shared.sh`
  
4. Average last three model checkpoints

   `average_models.sh`
   
5. Evaluate the final average NMT model on the test set with BLEU score

    `evaluate.sh`

### Alignment
Second, to train the word alignment model on the previously generated tokenised train sets 
`train.tok.en` and `train.tok.es` run the following script under 
the directory `src/tar/src/alignment`:

- `train_alignment_with_priors.sh`

### Translate and retrieve
Eventually, to generate the Spanish translation of the SQUAD datasets, both version v1.1 and v2.0,
run one of the following commands under the directory `src/tar/src/retrieve`

1. Download the SQuAD dataset, both version v1.1 and v2.0

    `download_squad.sh`

2. Generate the full dataset translation, called train-es:

    `translate_squad.sh <squad_file> -answers_from_alignment`
    
3. Optionally, generate the small dataset translation, called train-es-small:

    `translate_squad.sh <squad_file>`

The option 2 is used to generate the train-es datasets with almost 100% of the original SQuAD data while the
option 3 is used to generate the smaller train-es-small dataset, with about half of the original SQuAD data.

### The SQuAD-es datasets
Here, some statistics of the translated Spanish datasets showing the number of translated (context, question, anwers)
in the SQuAD-es datasets over the (context, question, anwers) in the SQuAD dataset.

|          | SQuAD-es v1.1  |     SQuAD-es v2.0  |
| ------------- | ------------- | ------------- |
|train-es| 87595/87599  | 46260/87599 |
|train-es-small| 130313/130319  | 69202/130319  |

For both the SQuAD version 1.1 and 2.0, the train-es version datasets contain almost 100% of the original SQuAD 
examples while the train-es-small datasets contains about the half of the total SQuAD examples. However, 
the bigger train-es datasets contain more noisy examples, 
in terms of wrongly translated (context, question, answer) triples, compared to the smaller train-es-small,

To better understand the difference between them, read the error analysis on the SQuAD-es v1.1 dataset carried out 
in the referenced paper.

## Question answering
This section shows how to train a question answering (QA) system with the SQuAD-es training dataset by fine-tuning 
a Multilingual-BERT (mBERT) model. The evaluation is performed on the recently proposed 
[MLQA corpus](https://github.com/facebookresearch/MLQA) and [XQuAD corpus](https://github.com/deepmind/xquad) benchmarks for cross-lingual QA evaluation.

### Installation
All the python dependencies and other non-python libraries can be automatically installed
by running the script: `src/qa/setup_env.sh`

### Train
To train a QA system, run the following script under the directory `src/qa`:

   `train_m-bert.sh <squad_file> <squad_version>`

_squad_file_  is the SQuAD dataset

_squad_version_ is a string with the version, namely as v1 or v2.

The resulting model will be saved in the default directory `data/training/m-bert_<squad_file>`

### Evaluate

To evaluate the trained QA systems, run the following script under the directory `src/qa`:

1. Download the MLQA and XQuAD evaluation benchmark:

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
    
### Spanish QA performance with SQuAD-es v1.1
Here the F1 and Exact Match scores on both the MLQA and XQuAD benchmarks compared to the 
corresponding best Spanish baselines provided in both the [MLQA paper](https://arxiv.org/abs/1910.07475)
and the [XQuAD paper](https://arxiv.org/abs/1910.11856v1).

|   Model       |            F1 / EM (MLQA)  |     F1 / EM (XQuAD)  |
| ------------- | ------------- | ------------- |
|TAR-train + mBERT (train-v1.1-es) | **68.1** / 48.3  | **77.6 / 61.8**  |
|TAR-train + mBERT (train-v1.1-es_small) | 65.5 / 47.2  | 73.8 / 59.5  |
|-------------------------------------------------------|----------------|---------------- |
|XLM (MLM + TLM, 15 languages)| 68.0 /49.8| -|
|JointMulti 200k voc  | - |  74.3 / 55.3|

The table shows significant improvements over the baselines establishing the **new state-of-the-art**
for the Spanish QA task on both the F1 and EM scores on the XQuAD corpus.

## Reference
If you use the SQuAD-es and find it useful for your research, please cite:

    @misc{carrino2019automatic,
        title={Automatic Spanish Translation of the SQuAD Dataset for Multilingual Question Answering},
        author={Casimiro Pio Carrino and Marta R. Costa-jussà and José A. R. Fonollosa},
        year={2019},
        eprint={1912.05200},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    
## License
The SQuAD-es dataset is licensed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode) license
The code in this repository is licensed under the GNU GPLv3 license according to the LICENSE file.