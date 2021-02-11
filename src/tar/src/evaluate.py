# Script to compute the translation quality of TAR-translated dataset if a human translation reference is available
# In addition, the script will take care of the alignment by matching the question ids
import json
from bleu import list_bleu
import argparse
from ordered_set import OrderedSet
from tqdm import tqdm
from collections import defaultdict
import os


def remove_line_breaks(text):
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    return text

def content_to_qids(dataset, extract_titles=False):
    # Extract titles (optional), contexts, questions and answers.
    context_titles_to_qids = defaultdict(list)
    question_answers_to_qids = defaultdict(list)
    for data in tqdm(dataset['data'], 'Get paragraphs'):
        for par in data['paragraphs']:
            for qa in par['qas']:
                # excluding questions with multiple answers to place questions and answers at the same tree level
                if len(qa['answers']) == 1:
                    answer = qa['answers'][0]['text']
                    context_titles_to_qids[par['context']].append(qa['id'])
                    # both questions and answers are at same level and the same alignment strategy can be applied
                    question_answers_to_qids[qa['question']].append(qa['id'])
                    question_answers_to_qids[answer].append(qa['id'])

    if extract_titles:
        for data in tqdm(dataset['data'], 'Get paragraphs'):
            for par in data['paragraphs']:
                for qa in par['qas']:
                    # both title and context are at same level and the same alignment strategy can be applied
                    context_titles_to_qids[par['title']].append(qa['id'])

    return context_titles_to_qids, question_answers_to_qids


def align_content(reference_file, translation_file, extract_titles=False):
    # Load data
    with open(reference_file) as rf, open(translation_file) as tf:
        dataset_ref = json.load(rf)
        dataset_tra = json.load(tf)

    context_titles_to_qids_ref, question_answers_to_qids_ref = content_to_qids(dataset_ref)
    context_titles_to_qids_tra, question_answers_to_qids_tra = content_to_qids(dataset_tra)

    references = []
    translations = []
    # align context and titles (if extracted)
    for content_ref, qids_ref in context_titles_to_qids_ref.items():
        for content_tra, qids_tra in context_titles_to_qids_tra.items():
            # make sure contexts and titles share at least one question id
            if set(qids_tra).intersection(set(qids_ref)):
                references.append(remove_line_breaks(content_ref))
                translations.append(remove_line_breaks(content_tra))

    # # align question and answers
    # for content_ref, qids_ref in question_answers_to_qids_ref.items():
    #     for content_tra, qids_tra in question_answers_to_qids_tra.items():
    #         # make sure questions and answers have the same question id
    #         if qids_tra == qids_ref:
    #             references.append(content_ref)
    #             translations.append(content_tra)

    references = list(OrderedSet(references))
    translations = list(OrderedSet(translations))
    assert len(references) == len(translations), 'References and translations are not aligned!'
    with open(os.path.join(SCRIPT_DIR, 'references.txt'), 'w') as rf, open(os.path.join(SCRIPT_DIR, 'translations.txt'),
                                                                           'w') as tf:
        rf.writelines(f'{line}\n' for line in references)
        tf.writelines(f'{line}\n' for line in translations)
    return references, translations


def bleu(references, translations):
    score = list_bleu([references], translations)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--translation_file', type=str, help='File with translated SQUAD data')
    parser.add_argument('--reference_file', type=str, help='File with translated SQUAD data')
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    refs, trans = align_content(args.reference_file, args.translation_file)

    print('BLEU = ', bleu(refs, trans))
