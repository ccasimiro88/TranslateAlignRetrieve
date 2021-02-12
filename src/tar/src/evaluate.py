# Script to compute the translation quality of TAR-translated dataset if a human translation reference is available
# In addition, the script will take care of the alignment by matching the question ids
import json
from bleu import list_bleu
import argparse
from ordered_set import OrderedSet
from tqdm import tqdm
from collections import defaultdict
import os
import logging


def remove_line_breaks(text):
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    return text


def content_to_qids(dataset):
    # Extract contexts, questions and answers.
    # Title are not extracted since they are not used as training inputs
    context_to_qids = defaultdict(list)
    questions_to_qids = defaultdict(list)
    answers_to_qids = defaultdict(list)
    for data in tqdm(dataset['data'], 'Get paragraphs'):
        for par in data['paragraphs']:
            for qa in par['qas']:
                # excluding questions with multiple answers to place questions and answers at the same tree level
                if len(qa['answers']) == 1:
                    answer = qa['answers'][0]['text']
                    context_to_qids[remove_line_breaks(par['context'])].append(qa['id'])
                    # both questions and answers are at same level and the same alignment strategy can be applied
                    questions_to_qids[qa['question']].append(qa['id'])
                    answers_to_qids[answer].append(qa['id'])

    return context_to_qids, questions_to_qids, answers_to_qids


def align_content(reference_file, translation_file):
    # Load data
    with open(reference_file) as rf, open(translation_file) as tf:
        dataset_ref = json.load(rf)
        dataset_tra = json.load(tf)

    context_to_qids_ref, questions_to_qids_ref, answers_to_qids_ref = content_to_qids(dataset_ref)
    context_to_qids_tra, questions_to_qids_tra, answers_to_qids_tra = content_to_qids(dataset_tra)

    references = []
    translations = []
    # # align context
    for content_ref, qids_ref in context_to_qids_ref.items():
        for content_tra, qids_tra in context_to_qids_tra.items():
            # make sure contexts and titles share at least one question id
            if set(qids_tra).intersection(set(qids_ref)):
                references.append(content_ref)
                translations.append(content_tra)
                logging.info(f'REF: {content_ref} ||| TRA: {content_tra}')

    # align question and answers
    for content_ref, qids_ref in questions_to_qids_ref.items():
        for content_tra, qids_tra in questions_to_qids_tra.items():
            # make sure questions and answers have the same question id
            if qids_tra == qids_ref:
                references.append(content_ref)
                translations.append(content_tra)
                logging.info(f'REF: {content_ref} ||| TRA: {content_tra}')

    for content_ref, qids_ref in answers_to_qids_ref.items():
        for content_tra, qids_tra in answers_to_qids_ref.items():
            # make sure questions and answers have the same question id
            if qids_tra == qids_ref:
                references.append(content_ref)
                translations.append(content_tra)
                logging.info(f'REF: {content_ref} ||| TRA: {content_tra}')

    references = list(OrderedSet(references))
    translations = list(OrderedSet(translations))
    return references, translations


def bleu(references, translations, detokenize):
    score = list_bleu([references], translations, detok=detokenize)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--translation_file', type=str, help='File with automatic translated SQUAD data')
    parser.add_argument('--reference_file', type=str, help='File with human translated SQUAD data')
    parser.add_argument('--detokenize', default=True, type=str,
                        help='Detokenize file before to compute BLEU score (recommended)')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir = args.output_dir if args.output_dir else os.path.dirname(os.path.realpath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    references, translations = align_content(args.reference_file, args.translation_file)
    assert len(references) == len(translations), 'References and translations are not aligned!'

    # Write references and translations to files
    with open(os.path.join(output_dir, 'references.txt'), 'w') as rf, \
            open(os.path.join(output_dir, 'translations.txt'), 'w') as tf:
        rf.writelines(f'{line}\n' for line in references)
        tf.writelines(f'{line}\n' for line in translations)

    score = bleu(references, translations, args.detokenize)
    logging.info(f'BLEU = {score}')
    with open(os.path.join(output_dir, 'bleu.txt'), 'w') as bf:
        bf.write(f'BLEU = {score}')

