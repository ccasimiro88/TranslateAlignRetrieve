import json
import time
from nltk import sent_tokenize
import subprocess
import csv
from tqdm import tqdm
import os
from collections import defaultdict
import pickle
import argparse
import translate_squad_utils as utils
from tqdm import tqdm


class SquadTranslator:
    def __init__(self,
                 squad_file,
                 lang_source,
                 lang_target,
                 output_dir,
                 use_translation_service,
                 alignment_type,
                 retrieve_answers_from_alignment,
                 batch_size):

        self.squad_file = squad_file
        self.lang_source = lang_source
        self.lang_target = lang_target
        self.output_dir = output_dir
        self.use_translation_service = use_translation_service
        self.alignment_type = alignment_type
        self.retrieve_answers_from_alignment = retrieve_answers_from_alignment
        self.batch_size = batch_size

        # initialize content_translated_alignments
        self.content_translated_alignment = defaultdict()

        # Translate all the textual content in the SQUAD dataset,
        # that are, context, questions and answers. The alignment between
        # context and its translation is then computed. The output is a
        # dictionary with context, question, answer as keys and their
        # translation/alignment as values

    def translate_align_squad_content(self, tokenized):

        # Check is the content of SQUAD has been translated and aligned already
        context_sentences_translations_alignments_filename = \
            os.path.join(self.output_dir,
                         '{}_content_translations_alignments.{}'.format(os.path.basename(self.squad_file),
                                                                        self.lang_target))
        if not os.path.isfile(context_sentences_translations_alignments_filename):
            # Load squad content and get squad contexts
            with open(self.squad_file) as hn:
                content = json.load(hn)
            # Extract contexts, questions and answers. The context is further
            # divided into sentence in order to translate and compute the alignment.
            titles = [data['title']
                      for data in tqdm(content['data'])]
            context_sentences = [context_sentence
                                 for data in tqdm(content['data'])
                                 for paragraph in tqdm(data['paragraphs'])
                                 for context_sentence in
                                 tqdm(sent_tokenize(utils.remove_line_breaks(paragraph['context']),
                                                    language='english'))
                                 if context_sentence]

            questions = [qa['question']
                         for data in tqdm(content['data'])
                         for paragraph in tqdm(data['paragraphs'])
                         for qa in paragraph['qas']
                         if qa['question']]

            answers = [answer['text']
                       for data in tqdm(content['data'])
                       for paragraph in tqdm(data['paragraphs'])
                       for qa in paragraph['qas']
                       for answer in qa['answers']
                       if answer['text']]

            # extract plausible answers when 'is_impossible == True'
            plausible_answers = []
            for data in tqdm(content['data']):
                for paragraph in tqdm(data['paragraphs']):
                    for qa in paragraph['qas']:
                        if qa['is_impossible']:
                            for answer in qa['plausible_answers']:
                                plausible_answers.append(answer['text'])

            content = titles + context_sentences + questions + answers + plausible_answers

            # Remove duplicated while keeping the order of occurrence
            content = sorted(set(content), key=content.index)

            # Translate contexts, questions and answers all together and write to file.
            # Also remove duplicates before to translate with set
            if self.use_translation_service:
                content_translated = [utils.translate(sentence) for sentence in tqdm(content)]
            else:
                content_translated = utils.translate_script(content, self.output_dir, self.batch_size)

            # Compute alignments
            context_sentence_questions_answers_alignments = \
                utils.compute_efolmal_sentence_alignment(content,
                                                         self.lang_source,
                                                         content_translated,
                                                         self.lang_target,
                                                         self.alignment_type,
                                                         self.output_dir,
                                                         alignment_tokenized=True)

            # Add translations and alignments
            for sentence, sentence_translated, alignment in zip(content,
                                                                content_translated,
                                                                context_sentence_questions_answers_alignments):
                self.content_translated_alignment[sentence] = {'translation': sentence_translated,
                                                               'alignment': alignment}
            with open(context_sentences_translations_alignments_filename, 'wb') as fn:
                pickle.dump(self.content_translated_alignment, fn)

        # Load content translated and aligned from file
        else:
            with open(context_sentences_translations_alignments_filename, 'rb') as fn:
                self.content_translated_alignment = pickle.load(fn)

    # Parse the SQUAD file and replace the questions, context and answers field with their translations
    # using the content_translated_alignment

    # For the answer translated the following two-steps logic is applied:
    # 1) Translate the answer and find them in the context translated
    #   1.1) searching around the answer start provided by the alignment
    #   1.2) if not the previous, searching from the beginning of the context translated
    #
    # 2) If the previous two steps fail, optionally extract the answer from the context translated
    # using the answer start and answer end provided by the alignment
    # TODO: clean the SQUAD datasets during translation to avoid run twice over the content
    def translate_squad(self, tokenized):

        with open(self.squad_file) as fn:
            content = json.load(fn)

        for data in tqdm(content['data']):
            title = data['title']
            data['title'] = self.content_translated_alignment[title]['translation']
            for paragraphs in tqdm(data['paragraphs']):
                context = paragraphs['context']

                context_sentences = [s for s in sent_tokenize(utils.remove_line_breaks(context), language='english')]
                context_translated = ' '.join(self.content_translated_alignment[s]['translation']
                                              for s in context_sentences)
                context_alignment_tok = utils.compute_context_alignment(
                    [self.content_translated_alignment[s]['alignment']
                     for s in context_sentences])

                # Translate context and replace its value back in the paragraphs
                paragraphs['context'] = context_translated
                for qa in tqdm(paragraphs['qas']):
                    question = qa['question']
                    question_translated = self.content_translated_alignment[question]['translation']
                    qa['question'] = question_translated

                    # Translate normal answers
                    if not qa['is_impossible']:
                        for answer in tqdm(qa['answers']):
                            answer_translated = self.content_translated_alignment[answer['text']]['translation']
                            answer_translated, answer_translated_start = \
                                utils.extract_answer_translated(answer,
                                                                answer_translated,
                                                                context,
                                                                context_translated,
                                                                context_alignment_tok,
                                                                self.retrieve_answers_from_alignment)
                            answer['text'] = answer_translated
                            answer['answer_start'] = answer_translated_start

                    else:
                        # Translate plausible answers
                        for plausible_answer in tqdm(qa['plausible_answers']):
                            plausible_answer_translated = self.content_translated_alignment[plausible_answer['text']]['translation']
                            answer_translated, answer_translated_start = \
                                utils.extract_answer_translated(plausible_answer,
                                                                plausible_answer_translated,
                                                                context,
                                                                context_translated,
                                                                context_alignment_tok,
                                                                self.retrieve_answers_from_alignment)
                            plausible_answer['text'] = answer_translated
                            plausible_answer['answer_start'] = answer_translated_start

        print('Cleaning...')
        # Clean translated content from empty answers
        # Parse the file and create a cleaned copy
        content_translated = content
        content_cleaned = {'version': content['version'], 'data': []}
        total_answers = 0
        total_correct_answers = 0
        for idx_data, data in tqdm(enumerate(content_translated['data'])):
            content_title = content_translated['data'][idx_data]['title']
            content_cleaned['data'].append({'title': content_title, 'paragraphs': []})
            for par in tqdm(data['paragraphs']):
                qas_cleaned = []
                for idx_qa, qa in tqdm(enumerate(par['qas'])):
                    question = qa['question']
                    if not qa['is_impossible']:
                        correct_answers = []
                        for a in qa['answers']:
                            total_answers += 1
                            if a['text']:
                                total_correct_answers += 1
                                correct_answers.append(a)
                        correct_plausible_answers = []
                    else:
                        correct_plausible_answers = []
                        for pa in qa['plausible_answers']:
                            total_answers += 1
                            if pa['text']:
                                total_correct_answers += 1
                                correct_plausible_answers.append(pa)
                        correct_answers = []

                    # add data to content cleaned
                    # answers
                    if correct_answers:
                        content_qas_id = qa['id']
                        content_qas_is_impossible = qa['is_impossible']
                        correct_answers_from_context = []
                        for a in qa['answers']:
                            start = a['answer_start']
                            correct_answers_from_context.append(
                                {'text': par['context'][start:start + len(a['text'])],
                                 'answer_start': start})
                        qa_cleaned = {'question': question,
                                      'answers': correct_answers_from_context,
                                      'id': content_qas_id,
                                      'is_impossible': content_qas_is_impossible}
                        qas_cleaned.append(qa_cleaned)
                    if correct_plausible_answers and not correct_answers:
                        content_qas_id = qa['id']
                        content_qas_is_impossible = qa['is_impossible']
                        correct_answers_from_context = []
                        for a in qa['answers']:
                            start = a['answer_start']
                            correct_answers_from_context.append(
                                {'text': par['context'][start:start + len(a['text'])],
                                 'answer_start': start})
                        qa_cleaned = {'question': question,
                                      'answers': correct_answers,
                                      'plausible_answers': correct_plausible_answers,
                                      'id': content_qas_id,
                                      'is_impossible': content_qas_is_impossible}
                        qas_cleaned.append(qa_cleaned)

                # Add the paragraph only if there are non-empty question-answer examples inside
                if qas_cleaned:
                    content_context = par['context']
                    content_cleaned['data'][idx_data]['paragraphs'].append(
                        {'context': content_context, 'qas': qas_cleaned})

        # Write the content back to the translated dataset
        if self.retrieve_answers_from_alignment:
            content_translated_file = os.path.join(self.output_dir,
                                                   os.path.basename(self.squad_file) +
                                                   '_retrieve_answers_alignment_{}'.format(self.lang_target))
        else:
            content_translated_file = os.path.join(self.output_dir,
                                                   os.path.basename(self.squad_file) + '_{}'.format(self.lang_target))

        with open(content_translated_file, 'w') as fn:
            json.dump(content_cleaned, fn)

        accuracy = round((total_correct_answers / total_answers) * 100, 2)
        print('{}: Percentage of translated examples'
              ' (correct answers/total answers): {}/{} = {}%'.format(os.path.basename(content_translated_file),
                                                                     total_correct_answers, total_answers,
                                                                     accuracy))


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-squad_file', type=str, help='SQUAD dataset to translate')
    parser.add_argument('-lang_source', type=str, help='language of the SQUAD dataset to translate')
    parser.add_argument('-lang_target', type=str, help='translation language')
    parser.add_argument('-output_dir', type=str, help='directory where all the generated files are stored')
    parser.add_argument('-retrieve_answers_from_alignment', action='store_true',
                        help='retrieve translated answers only from the alignment')
    parser.add_argument('-use_translation_service', action='store_true', help='use a given translation service')
    parser.add_argument('-alignment_type', type=str, default='forward', help='use a given translation service')
    parser.add_argument('-alignment_tokenized', action='store_true', help='Compute alignment on tokenized text')
    parser.add_argument('-batch_size', type=int, default='32', help='batch_size for the translation script '
                                                                    '(change this value in case of CUDA out-of-memory')
    args = parser.parse_args()

    # Create output directory if doesn't exist already
    try:
        os.mkdir(args.output_dir)
    except FileExistsError:
        pass

    translator = SquadTranslator(args.squad_file,
                                 args.lang_source,
                                 args.lang_target,
                                 args.output_dir,
                                 args.use_translation_service,
                                 args.alignment_type,
                                 args.retrieve_answers_from_alignment,
                                 args.batch_size)

    print('Translate SQUAD textual content and compute alignments...')
    translator.translate_align_squad_content(args.alignment_tokenized)

    print('Create translated SQUAD...')
    translator.translate_squad(args.alignment_tokenized)

    end = time.time()
    print('Total execution time: {}'.format(end - start))
