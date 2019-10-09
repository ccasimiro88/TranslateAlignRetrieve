## Translate QA datasets

The aim of this repo is to create scripts to parse and translate the Question-Answering dataset like 
SQUAD and Natural Questions from English to another language. 
It is based on two main components:
 1) A translation model from English to the target language
 2) An alignment model from English to the target language
 
Currently, there is a script for translating SQUAD from English to Spanish.

### Install
The project requires both python and non-python dependencies. All the necessary libraries can be installed
by running the following script:
 `./setup.sh` 
 
### Usage
To translate the SQUAD dataset just run the command:
`python ./translate_squad.py -squad_file <squad_file> \
                           -language_source <language_source> 
                           -language_target <language_target>`

Optionally, two more options are available:

 - `-retrieve_answers_from_alignment`
 
    This option enable the retrieval of the translated answers by fully relying on the alignment model
    
 - `-priors_alignment_file <priors_alignment_file>`
 
    This option use a pre-trained alignment model to infer the alignment on the documents in
    the SQUAD file. In general, he priors are computed from a parallel corpus than is
    bigger than the total number of documents in the SQUAD in order to get a more accurate alignment model.
    The following script generates those priors: `./get_priors_alignments_eflomal.sh <file_source> <file_target> <language_target>`

### Implementation logic to translate the SQUAD dataset
This section describes the logic used to retrieve the corresponding answer translated from the document translated.

#### Document extraction
1. Extract documents in the dataset split them into sentences
2. Translate these sentences with an external translation service
3. Write to file the sentences and their translations
4. Create a dictionary with the documents as keys and the sentences and their translations as values

#### Aligment
1. Compute the alignments between the document sentences and their translations
2. Create a dictionary with the documents as keys and the sentences translated and the previous alignment 

#### Answer translated retrieval
For each answer and a document in the dataset, we compute the answer translated following the logic:

Retrieve the answer translated in the document translated

1. Translate the answer and the document
2. Use the alignment to get the index where the answer translated starts in the document translated
3. Find the translated answer translated starting from the previous index
4. If no match is found in 1.3, find the answer translated from the beginning of the document translated
5. If no match is found in 1.4, extract the answer translated directly from the document translated 
    using the start and end position from the alignment 

The point 5 is the most critical because if the alignment is not accurate, the retrieved answer translated
is not good. That is why this behaviour is optional (see `-retrieve_answers_from_alignment`)