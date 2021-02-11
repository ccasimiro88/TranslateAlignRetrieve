import json
import time
import subprocess
import csv
from tqdm import tqdm
import os
from collections import defaultdict
import pickle
import argparse
import translate_squad_utils as utils
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import torch
import logging
import random
from ordered_set import OrderedSet
import tempfile


class Aligner:
    def __init__(self, alignment_model):
        self.alignment_model = alignment_model

    # TODO: implement priors for eflomal
    @staticmethod
    def eflomal(source_sentences, source_lang, translated_sentences, target_lang,
                alignment_type, file, output_dir):
        filename = os.path.basename(file)
        source_sentences = [utils.tokenize(sentence, source_lang) for sentence in source_sentences]
        translated_sentences = [utils.tokenize(sentence, target_lang) for sentence in translated_sentences]

        source_filename = os.path.join(output_dir, f'cached_{filename}_source_align')
        with open(source_filename, 'w') as sf:
            sf.writelines('\n'.join(s for s in source_sentences))

        translation_filename = os.path.join(output_dir, f'cached_{filename}_target_align')
        with open(translation_filename, 'w') as tf:
            tf.writelines('\n'.join(s for s in translated_sentences))

        # TODO: add the case with priors
        alignment_filename = os.path.join(output_dir, f'cached_{filename}_alignment')
        efolmal_cmd = SCRIPT_DIR + f'/../alignment/compute_alignment.sh ' \
                                   f'{source_filename} {source_lang} {translation_filename} {target_lang}' \
                                   f' {alignment_type} {alignment_filename}'
        subprocess.run(efolmal_cmd.split())

        with open(alignment_filename) as af:
            alignments = [a.strip() for a in af.readlines()]

        # os.remove(source_filename)
        # os.remove(translation_filename)
        # os.remove(alignment_filename)
        return alignments

    def align(self, *alignment_args):
        if self.alignment_model == 'eflomal':
            logging.info(f'Using {self.alignment_model} alignment method')
            return self.eflomal(*alignment_args)

        else:
            raise NotImplementedError('Unsupported alignment model!')
        pass


class Translator:
    def __init__(self, translation_engine):
        self.translation_engine = translation_engine

    # Translate via the OpenNMT-py script
    @staticmethod
    def opennmt_en_es(source_sentences, batch_size):
        # filename = os.path.basename(file)
        # source_filename = os.path.join(output_dir, f'{filename}_source_translate')
        source_filename = tempfile.NamedTemporaryFile().name
        translation_filename = tempfile.NamedTemporaryFile().name
        with open(source_filename, 'w') as sf:
            sf.writelines('\n'.join(s for s in source_sentences))

        # translation_filename = os.path.join(output_dir, f'{filename}_target_translated')
        en2es_translate_cmd = SCRIPT_DIR + f'/../nmt/en2es_translate.sh ' \
                                           f'{source_filename} {translation_filename} {batch_size}'
        subprocess.run(en2es_translate_cmd.split())

        with open(translation_filename) as tf:
            translated_sentences = [s.strip() for s in tf.readlines()]

        os.remove(source_filename)
        os.remove(translation_filename)
        return translated_sentences

    # Translate with MariantMT with HuggingFace library
    @staticmethod
    def marianmt_hf(sentences, batch_size, lang_source, lang_target):
        # TODO: check if src-tgt model exists
        def chunks(sentences, batch_size):
            for i in range(0, len(sentences), batch_size):
                yield sentences[i:i + batch_size]

        model_name = f'Helsinki-NLP/opus-mt-{lang_source}-{lang_target}'
        model = MarianMTModel.from_pretrained(model_name).to(DEVICE).half()
        logging.info(f"Using {model_name} MarianMT translator for {lang_source}-{lang_target}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        translations = []
        batches = list(chunks(sentences, batch_size))
        for chunk in tqdm(batches, desc=f'Translating with batch size {batch_size}'):
            batch = tokenizer.prepare_seq2seq_batch(src_texts=chunk, return_tensors='pt').to(DEVICE)
            translated = model.generate(**batch)
            translations.extend([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
        return translations

    def translate(self, sentences, lang_source, lang_target, batch_size, lang_pivot):
        if self.translation_engine == 'marianmt_hf':
            # Translation through pivot language
            if lang_pivot:
                logging.info(f"Using Back-translation with pivot language {lang_pivot}")
                return self.marianmt_hf(self.marianmt_hf(sentences, lang_source, lang_pivot, batch_size),
                                        lang_pivot, lang_target, batch_size)
            else:
                return self.marianmt_hf(sentences, lang_source, lang_target, batch_size)

        elif self.translation_engine == 'opennmt_en-es':
            assert (lang_source == 'en' and lang_target == 'es'), \
                ValueError('opennmt translator only supported for en-es')
            logging.info(f"Using OpenNMT-py translator for {lang_source}-{lang_target}")
            return self.opennmt_en_es(sentences, batch_size)

        else:
            raise NotImplementedError('Unsupported translation engine!')


class SquadTranslator:
    def __init__(self,
                 squad_file,
                 lang_source,
                 lang_target,
                 output_dir,
                 alignment_type,
                 answers_from_alignment,
                 batch_size):

        self.squad_file = squad_file
        self.lang_source = lang_source
        self.lang_target = lang_target
        self.output_dir = output_dir
        self.alignment_type = alignment_type
        self.answers_from_alignment = answers_from_alignment
        self.batch_size = batch_size

        self.squad_version = None
        self.dataset = None
        self.content_translations_alignments = defaultdict()

    @staticmethod
    def load_dataset(dataset_file, sample_size):
        with open(dataset_file) as hn:
            dataset = json.load(hn)
        if sample_size:
            logging.info(f'Sampling {args.sample_size} articles from {args.squad_file}')
            dataset = {'version': dataset['version'],
                       'data': random.sample(dataset['data'], sample_size)}
        return dataset

    def extract_content(self, dataset):
        # Extract contexts, questions and answers. The context is further
        # divided into sentence in order to translate and compute the alignment.
        # titles = [data['title']
        #           for data in tqdm(dataset['data'], desc='Get Titles')]
        context_sentences = [context_sentence
                             for data in tqdm(dataset['data'], 'Get paragraphs')
                             for paragraph in data['paragraphs']
                             for context_sentence in
                             tqdm(utils.tokenize_sentences(utils.remove_line_breaks(paragraph['context']),
                                                           lang=self.lang_source))
                             if context_sentence]

        questions = [qa['question']
                     for data in tqdm(dataset['data'], desc='Get Questions')
                     for paragraph in data['paragraphs']
                     for qa in paragraph['qas']
                     if qa['question']]

        answers = [answer['text']
                   for data in tqdm(dataset['data'], desc='Get answers')
                   for paragraph in data['paragraphs']
                   for qa in paragraph['qas']
                   for answer in qa['answers']
                   if answer['text']]

        # extract plausible answers when 'is_impossible == True' for SQUAD v2.0
        if self.squad_version == 'v2.0':
            plausible_answers = []
            for data in tqdm(dataset['data']):
                for paragraph in tqdm(data['paragraphs']):
                    for qa in paragraph['qas']:
                        if qa['is_impossible']:
                            for answer in qa['plausible_answers']:
                                plausible_answers.append(answer['text'])
        else:
            plausible_answers = []

        content = context_sentences + questions + answers + plausible_answers

        # Remove duplicated while keeping the order of occurrence
        # content = sorted(set(content), key=content.index)
        content = list(OrderedSet(content))
        return content

    # Translate all the textual content in the SQUAD dataset, that are, context, questions and answers.
    # The alignment between context and its translation is then computed.
    # The output is a dictionary with context, question, answer as keys and their translation/alignment as values
    # TODO: use some class attributes as funcion arguments
    def translate_align_content(self, overwrite_cached_data, sample_size, translation_engine, lang_pivot,
                                alignment_model):
        # Load squad content and get squad contexts
        dataset = self.dataset = self.load_dataset(self.squad_file, sample_size)

        # Get SQuAD version
        self.squad_version = dataset['version']

        # Check is the content of SQUAD has been translated and aligned already
        content_translations_alignments_file = os.path.join(self.output_dir,
                                                            'cached_{}_content_translations_alignments_{}.pickle'.format(
                                                                os.path.basename(self.squad_file), self.lang_target))
        if not os.path.isfile(content_translations_alignments_file) or overwrite_cached_data:
            # Extract content
            content = self.extract_content(dataset)

            # Translate
            translator = Translator(translation_engine=translation_engine)
            content_translated = translator.translate(content,
                                                      self.batch_size,
                                                      self.lang_source,
                                                      self.lang_target,
                                                      lang_pivot)
            # if translation_engine == 'marianmt_hf':
            #     translator = Translator(translation_engine=translation_engine)
            #     # Use MarianMT pre-trained translation engines:
            #     content_translated = translator.translate(content,
            #                                               self.lang_source,
            #                                               self.lang_target,
            #                                               self.batch_size,
            #                                               lang_pivot)
            # elif translation_engine == 'opennmt_en-es':
            #     if self.lang_target == 'es':
            #         # Translate contexts, questions and answers all together and write to file.
            #         # Also remove duplicates before to translate with set
            #         content_translated = utils.opennmt_en_es(content, self.squad_file, self.output_dir, self.batch_size)
            # else:
            #     raise ValueError(f"Unsupported translation engine {translation_engine}")

            # Compute alignments
            assert content_translated, 'Content translation is empty!'
            aligner = Aligner(alignment_model=alignment_model)
            context_sentence_questions_answers_alignments = aligner.align(content,
                                                                          self.lang_source,
                                                                          content_translated,
                                                                          self.lang_target,
                                                                          self.alignment_type,
                                                                          self.squad_file,
                                                                          self.output_dir)
            assert context_sentence_questions_answers_alignments, 'Alignment is empty!'

            # Add translations and alignments
            for sentence, sentence_translated, alignment in zip(content,
                                                                content_translated,
                                                                context_sentence_questions_answers_alignments):
                self.content_translations_alignments[sentence] = {'translation': sentence_translated,
                                                                  'alignment': alignment}
            with open(content_translations_alignments_file, 'wb') as fn:
                pickle.dump(self.content_translations_alignments, fn)

        # Load content translated and aligned from file
        else:
            logging.info('Using cached data previously computed (translations and alignments')
            with open(content_translations_alignments_file, 'rb') as fn:
                self.content_translations_alignments = pickle.load(fn)

    # Parse the SQUAD file and replace the questions, context and answers field with their translations
    # using the content_translations_alignments

    # For the answer translated the following two-steps logic is applied:
    # 1) Translate the answer and find them in the context translated
    #   1.1) searching around the answer start provided by the alignment
    #   1.2) if not the previous, searching from the beginning of the context translated
    #
    # 2) If the previous two steps fail, optionally extract the answer from the context translated
    # using the answer start and answer end provided by the alignment
    def translate_retrieve(self):
        dataset = self.dataset

        for data in tqdm(dataset['data']):
            # title = data['title']
            # data['title'] = self.content_translations_alignments[title]['translation']
            for paragraphs in data['paragraphs']:
                context = paragraphs['context']

                context_sentences = [s for s in utils.tokenize_sentences(utils.remove_line_breaks(context),
                                                                         lang=self.lang_source)]

                context_translated = ' '.join(self.content_translations_alignments[s]['translation']
                                              for s in context_sentences)
                context_alignment_tok = utils.compute_context_alignment(
                    [self.content_translations_alignments[s]['alignment']
                     for s in context_sentences])

                # Translate context and replace its value back in the paragraphs
                paragraphs['context'] = context_translated
                for qa in paragraphs['qas']:
                    question = qa['question']
                    question_translated = self.content_translations_alignments[question]['translation']
                    qa['question'] = question_translated

                    # Translate answers and plausible answers for SQUAD v2.0
                    if self.squad_version == 'v2.0':
                        if not qa['is_impossible']:
                            for answer in qa['answers']:
                                answer_translated = self.content_translations_alignments[answer['text']]['translation']
                                answer_translated, answer_translated_start = \
                                    utils.extract_answer_translated(answer,
                                                                    answer_translated,
                                                                    context,
                                                                    context_translated,
                                                                    context_alignment_tok,
                                                                    self.answers_from_alignment)
                                answer['text'] = answer_translated
                                answer['answer_start'] = answer_translated_start

                        else:
                            for plausible_answer in qa['plausible_answers']:
                                plausible_answer_translated = \
                                    self.content_translations_alignments[plausible_answer['text']]['translation']
                                answer_translated, answer_translated_start = \
                                    utils.extract_answer_translated(plausible_answer,
                                                                    plausible_answer_translated,
                                                                    context,
                                                                    context_translated,
                                                                    context_alignment_tok,
                                                                    self.answers_from_alignment)
                                plausible_answer['text'] = answer_translated
                                plausible_answer['answer_start'] = answer_translated_start

                    # Translate answers for SQUAD v1.1
                    else:
                        for answer in qa['answers']:
                            answer_translated = self.content_translations_alignments[answer['text']]['translation']
                            answer_translated, answer_translated_start = \
                                utils.extract_answer_translated(answer,
                                                                answer_translated,
                                                                context,
                                                                context_translated,
                                                                context_alignment_tok,
                                                                self.answers_from_alignment)
                            answer['text'] = answer_translated
                            answer['answer_start'] = answer_translated_start

        logging.info('Cleaning and refinements...')
        # Parse the file, create a copy of the translated version and clean it from empty answers
        content_translated = dataset
        content_cleaned = {'version': dataset['version'], 'data': []}
        total_answers = 0
        total_correct_plausible_answers = 0
        total_correct_answers = 0
        for idx_data, data in tqdm(enumerate(content_translated['data'])):
            content_title = content_translated['data'][idx_data]['title']
            content_cleaned['data'].append({'title': content_title, 'paragraphs': []})
            for par in data['paragraphs']:
                qas_cleaned = []
                for idx_qa, qa in enumerate(par['qas']):
                    question = qa['question']

                    # Extract answers and plausible answers for SQUAD v2.0
                    if self.squad_version == 'v2.0':
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
                                    total_correct_plausible_answers += 1
                                    correct_plausible_answers.append(pa)
                            correct_answers = []

                        # add answers and plausible answers to the dataset cleaned
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

                    # Extract answers for SQUAD v1.0
                    else:
                        correct_answers = []
                        for a in qa['answers']:
                            total_answers += 1
                            if a['text']:
                                total_correct_answers += 1
                                correct_answers.append(a)

                        # add answers and plausible answers to the dataset cleaned
                        if correct_answers:
                            content_qas_id = qa['id']
                            correct_answers_from_context = []
                            for a in qa['answers']:
                                start = a['answer_start']
                                correct_answers_from_context.append(
                                    {'text': par['context'][start:start + len(a['text'])],
                                     'answer_start': start})
                            qa_cleaned = {'question': question,
                                          'answers': correct_answers_from_context,
                                          'id': content_qas_id}
                            qas_cleaned.append(qa_cleaned)

                # Add the paragraph only if there are non-empty question-answer examples inside
                if qas_cleaned:
                    content_context = par['context']
                    content_cleaned['data'][idx_data]['paragraphs'].append(
                        {'context': content_context, 'qas': qas_cleaned})

        # Write the dataset back to the translated dataset
        if self.answers_from_alignment:
            translated_file = os.path.join(self.output_dir,
                                           os.path.basename(self.squad_file).replace(f'.json',
                                                                                     f'-{self.lang_target}.json'))
        else:
            translated_file = os.path.join(self.output_dir,
                                           os.path.basename(self.squad_file).replace(f'.json',
                                                                                     f'-{self.lang_target}_small.json'))

        with open(translated_file, 'w') as fn:
            json.dump(content_cleaned, fn)

        # Count correct answers and plausible answers for SQUAD v2.0
        if self.squad_version == 'v2.0':
            total_correct = total_correct_answers + total_correct_plausible_answers
            accuracy = round((total_correct / total_answers) * 100, 2)
            logging.info(f'Translated dataset: {os.path.realpath(translated_file)}:\n'
                         f'Percentage of translated examples {total_correct}/{total_answers} = {accuracy}%\n'
                         f'No. of answers: {total_correct_answers}\n'
                         f'No. of plausible answers: {total_correct_plausible_answers}')

        # Count correct answers
        else:
            total_correct = total_correct_answers
            accuracy = round((total_correct / total_answers) * 100, 2)
            logging.info(f'Translated dataset: {os.path.realpath(translated_file)}:\n'
                         f'Percentage of translated examples {total_correct}/{total_answers} = {accuracy}%\n'
                         f'No. of answers: {total_correct_answers}')


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--squad_file', type=str, help='SQUAD dataset to translate')
    parser.add_argument('--lang_source', type=str, default='en',
                        help='language of the SQUAD dataset to translate (the default value set to English)')
    parser.add_argument('--lang_target', type=str, help='translation language')
    parser.add_argument('--output_dir', type=str, help='directory where all the generated files are stored')
    parser.add_argument('--answers_from_alignment', action='store_true',
                        help='retrieve translated answers only from the alignment')
    parser.add_argument('--overwrite_cached_data', action='store_true',
                        help='Overwrite pre-computed cached data (translation and alignments)')
    parser.add_argument('--alignment_type', type=str, default='forward',
                        help='Type of alignment used (forward, reverse or symmetric')
    parser.add_argument('--batch_size', type=int, default='8',
                        help='Translate data in batches with a given batch_size (change in case of memory errors')
    parser.add_argument('--seed', type=int, default='1', help='Seed for random data selections')
    parser.add_argument('--sample_size', type=int, help='Sampling N random articles to translate from input file')
    parser.add_argument('--translation_engine', type=str, default='marianmt_hf',
                        help='Select translation engines (marianmt_hf or opennmt_en-es')
    parser.add_argument('--alignment_model', type=str, default='eflomal',
                        help='Select the alignment model (supported only eflomal)')
    parser.add_argument('--lang_pivot', type=str, help='Use pivot language to perform back-translation')
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    DEVICE = torch.cuda.current_device()
    random.seed(args.seed)

    # Create output directory if doesn't exist already
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    squadtranslator = SquadTranslator(args.squad_file,
                                      args.lang_source,
                                      args.lang_target,
                                      args.output_dir,
                                      args.alignment_type,
                                      args.answers_from_alignment,
                                      args.batch_size)

    logging.info('Translate SQUAD textual content and compute alignments')
    squadtranslator.translate_align_content(args.overwrite_cached_data, args.sample_size, args.translation_engine,
                                            args.lang_pivot, args.alignment_model)

    logging.info('Translate and retrieve the SQUAD dataset')
    squadtranslator.translate_retrieve()

    end = time.time()
    logging.info(f'Total execution time: {round(end - start)} s')
