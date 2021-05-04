import json
import time
import subprocess
import csv
from tqdm import tqdm
import os
from collections import defaultdict
import argparse
import translate_squad_utils as utils
from transformers import MarianMTModel, MarianTokenizer
from sacremoses import MosesTokenizer, MosesDetokenizer
from tqdm import tqdm
import torch
import logging
import random
from ordered_set import OrderedSet
import sentence_splitter
import tempfile
import jieba


class Tokenizer:
    def __init__(self, lang):
        self.lang = lang
        self.max_sent_len = 10
        self.split_delimiter = ';'

    @staticmethod
    def moses_tokenizer(lang):
        return MosesTokenizer(lang=lang)

    @staticmethod
    def moses_detokenize(lang):
        return MosesDetokenizer(lang=lang)

    def tokenize(self, text, return_str=True):
        if self.lang != 'zh':
            return MosesTokenizer(lang=self.lang).tokenize(text, return_str=return_str, escape=False)
        else:
            return " ".join(jieba.cut(text, cut_all=False))  # accurate segmentation mode

    def detokenize(self, text, return_str=True):
        if not isinstance(text, list):
            text = text.split()
        if self.lang != 'zh':
            return self.moses_detokenize(self.lang).detokenize(text, return_str=return_str)
        else:
            return "".join(text)

    # Chunk sentences longer than a maximum number of words/tokens based on a delimiter character.
    # This option is used only for very long sentences to avoid shorter translation than the
    # original source length.
    # Note that the delimiter can't be a trailing character
    def split_sentences(self, text, delimiter, max_sent_len, tokenized=True):
        text_len = len(self.tokenize(text, return_str=True).split()) if tokenized else len(text.split())
        if text_len >= max_sent_len:
            delimiter_match = delimiter + ' '
            text_chunks = [chunk.strip() for chunk in text.split(delimiter_match) if chunk]
            # Add the delimiter lost during chunking
            text_chunks = [chunk + delimiter for chunk in text_chunks[:-1]] + [text_chunks[-1]]
            return text_chunks
        return [text]

    def tokenize_sentences(self, text):
        sentences = [chunk
                     for sentence in sentence_splitter.SentenceSplitter(language=self.lang).split(text)
                     for chunk in self.split_sentences(sentence,
                                                       delimiter=self.split_delimiter,
                                                       max_sent_len=self.max_sent_len)]
        return sentences


class AnswerRetriever:
    def __init__(self, answer_retrieval, tokenizer_src, tokenizer_tgt):
        self.answer_retrieval = answer_retrieval
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

    # Extract the answer translated from the context translated only using context alignment.
    # A series of heuristics are applied in order to extract the answer
    @staticmethod
    def retrieve_answer_from_alignment(answer_text, answer_start, context_translated, context_alignment):
        # Get the corresponding start and end char of the answer_translated in the context translated
        # First, get all the index positions for each word in the answer
        answer_words_positions = [answer_start]
        for word in answer_text.split():
            pos = answer_start + len(word) + 1
            answer_words_positions.append(pos)

        # Second, get all the corresponding index positions in the answer translated
        answer_translated_words_positions = []
        for idx, pos in enumerate(answer_words_positions):
            pos = utils.get_left_right_close_index(context_alignment.keys(), pos, type='left')
            answer_translated_words_positions.append(context_alignment[pos])

        # Then, detect the start and end position in the answer translated
        start, end = min(answer_translated_words_positions), max(answer_translated_words_positions)
        answer_translated_start = start
        answer_translated_end = end

        # Also, get the next start position to retrieve the answer until that index
        answer_next_start = answer_start + len(answer_text) + 1
        answer_next_start = utils.get_left_right_close_index(context_alignment.keys(), answer_next_start, type='right')
        answer_translated_next_start = context_alignment[answer_next_start]

        # Check if the answer_translated next start index is smaller than the and answer_translated_end index.
        # If so move to the next right index until is greater than the answer_translated_end
        while answer_translated_next_start <= answer_translated_end:
            answer_translated_next_start = utils.shift_value_index_alignment(answer_translated_next_start,
                                                                             context_alignment)
            # If the maximum index is at the end of the alignment map, change its value to the last character
            if answer_translated_next_start == -1:
                answer_translated_next_start = len(context_translated)

        # Extract answer translated from context translated with answer_start and answer_next_start
        # the answer_translated is a span from the min to max char index (
        answer_translated = context_translated[answer_translated_start:answer_translated_next_start]
        return answer_translated, answer_translated_start

    # Extract the answer from a given context
    def retrieve_answer(self, answer, answer_translated, context, context_translated, context_alignment_tok,
                        postprocess):
        # First, compute the src2tran_alignment_char
        context_alignment_char = utils.get_src2tran_alignment_char(self.tokenizer_src, self.tokenizer_tgt,
                                                                   context_alignment_tok, context, context_translated)

        answer_text = answer['text']
        answer_start = answer['answer_start']

        # Retrieve answer ONLY from alignment (very noisy)
        if self.answer_retrieval == 'alignment':
            answer_translated, answer_translated_start = \
                self.retrieve_answer_from_alignment(answer_text, answer_start, context_translated,
                                                    context_alignment_char)

        elif self.answer_retrieval in ['span', 'span_plus_alignment']:
            # 1.1)
            # Match the answer_translated in the context_translated starting from the left-close char index
            # of the answer_start in the context_alignment. When the answer_start is present in the
            # context_alignment the closest index is exactly the answer_start index
            answer_start = utils.get_left_right_close_index(list(context_alignment_char.keys()),
                                                            answer_start,
                                                            type='left')
            try:
                answer_translated_start = context_alignment_char[answer_start]
            except KeyError:
                answer_translated, answer_translated_start = '', -1
            # Find answer_start in the translated context by looking close to the answer_translated_char_start
            # Shifting the index by an additional 20 chars to the left in the case
            # the alignment is not precise enough (20 chars is more or less 2/3 words).
            # If the shifted answer_start is smaller than 0, we set it as zero to match from start errors
            shift_index = -20
            answer_translated_start_shifted = answer_translated_start + shift_index
            if answer_translated_start_shifted < 0:
                answer_translated_start_shifted = 0

            if context_translated.lower().find(answer_translated.lower(), answer_translated_start_shifted) != -1:

                answer_translated_start = context_translated.lower().find(answer_translated.lower(),
                                                                          answer_translated_start_shifted)
                answer_translated_end = answer_translated_start + len(answer_translated)
                answer_translated = context_translated[answer_translated_start: answer_translated_end]

            # 1.2) Find the answer_translated in the context_translated from the beginning of the text
            elif context_translated.lower().find(answer_translated.lower(),
                                                 answer_translated_start) == -1:
                if context_translated.lower().find(answer_translated.lower()) != -1:
                    answer_translated_start = context_translated.lower().find(answer_translated.lower())
                    answer_translated_end = answer_translated_start + len(answer_translated)
                    answer_translated = context_translated[answer_translated_start: answer_translated_end]

            # 2) If no answer is found and the alignment retrieval option is on,
            # retrieve the answer from the context translated using the
            # answer start and answer end provided by the alignment
            if not answer_translated_start != -1 and not answer_translated:
                if self.answer_retrieval == 'span_plus_alignment':
                    answer_translated, answer_translated_start = \
                        self.retrieve_answer_from_alignment(answer_text, answer_start, context_translated,
                                                            context_alignment_char)
                # No answer translated found
                elif self.answer_retrieval == 'span':
                    answer_translated = ''
                    answer_translated_start = -1

        # Post-process if the answer is not empty
        if postprocess and answer_translated:
            answer_translated = utils.post_process_answers_translated(answer_text, answer_translated)
        return answer_translated, answer_translated_start


class Aligner:
    def __init__(self, alignment_model, tokenizer_src, tokenizer_tgt):
        self.alignment_model = alignment_model
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

    # TODO: implement priors for eflomal
    def eflomal(self, source_sentences, source_lang, translated_sentences, target_lang,
                alignment_type, output_dir):
        source_sentences = [self.tokenizer_src.tokenize(sentence, source_lang) for sentence in source_sentences]
        translated_sentences = [self.tokenizer_tgt.tokenize(sentence, target_lang) for sentence in translated_sentences]

        source_filename = tempfile.NamedTemporaryFile(dir=output_dir, prefix="align.source").name
        with open(source_filename, 'w', encoding='utf8') as sf:
            sf.writelines("\n".join(s for s in source_sentences))

        translation_filename = tempfile.NamedTemporaryFile(dir=output_dir, prefix="align.target").name
        with open(translation_filename, 'w', encoding='utf8') as tf:
            tf.writelines("\n".join(s for s in translated_sentences))

        # TODO: add the case with priors
        alignment_filename = tempfile.NamedTemporaryFile(dir=output_dir, prefix="alignment").name
        efolmal_cmd = SCRIPT_DIR + f'/../alignment/compute_alignment.sh ' \
                                   f'{source_filename} {translation_filename}' \
                                   f' {alignment_type} {alignment_filename}'

        subprocess.run(efolmal_cmd.split())

        with open(alignment_filename) as af:
            alignments = [a.strip() for a in af.readlines()]

        os.remove(source_filename)
        os.remove(translation_filename)
        os.remove(alignment_filename)

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

    @staticmethod
    def replace_empty_translations(translations):
        return [translation if translation else 'none' for translation in translations]

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

    def translate(self, sentences, batch_size, lang_source, lang_target, lang_pivot):
        if self.translation_engine == 'marianmt_hf':
            # Translation through pivot language
            if lang_pivot:
                logging.info(f"Using Back-translation with pivot language {lang_pivot}")
                return self.replace_empty_translations(
                    self.marianmt_hf(
                        self.marianmt_hf(sentences, batch_size, lang_source, lang_pivot),
                        batch_size, lang_pivot, lang_target)
                )
            else:
                return self.replace_empty_translations(
                    self.marianmt_hf(sentences, batch_size, lang_source, lang_target)
                )

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
                 answer_retrieval,
                 batch_size,
                 tokenizer_src,
                 tokenizer_tgt):

        self.squad_file = squad_file
        self.lang_source = lang_source
        self.lang_target = lang_target
        self.output_dir = output_dir
        self.alignment_type = alignment_type
        self.answer_retrieval = answer_retrieval
        self.batch_size = batch_size
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.sent_tokenizer = Tokenizer(lang=self.lang_source)

        self.squad_version = None
        self.dataset = None

    @staticmethod
    def load_dataset(dataset_file, sample_size=None):
        with open(dataset_file) as hn:
            dataset = json.load(hn)
        if sample_size:
            logging.info(f'Sampling {sample_size} articles from {dataset_file}')
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
                             self.sent_tokenizer.tokenize_sentences(utils.remove_line_breaks(paragraph['context']))
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
        if dataset['version'] == 'v2.0':
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
    def translate_align_content(self, dataset, overwrite_cached_data, translation_engine, lang_pivot,
                                alignment_model):
        # Check is the content of SQUAD has been translated and aligned already
        content_translations_alignments_file = os.path.join(self.output_dir, '.cached/',
                                                            '{}_content_translations_alignments_{}-{}.json'.format(
                                                                os.path.basename(self.squad_file),
                                                                self.lang_source,
                                                                self.lang_target))

        content_translations_file = os.path.join(self.output_dir, '.cached/',
                                                 '{}_content_translations_{}-{}.json'.format(
                                                     os.path.basename(self.squad_file),
                                                     self.lang_source,
                                                     self.lang_target))

        os.makedirs(os.path.dirname(content_translations_alignments_file), exist_ok=True)
        # if not os.path.isfile(content_translations_alignments_file) or overwrite_cached_data:
        # Extract content
        content = self.extract_content(dataset)
        # Translate
        translator = Translator(translation_engine=translation_engine)

        if not os.path.isfile(content_translations_file) or overwrite_cached_data:
            content_translated = translator.translate(content,
                                                      self.batch_size,
                                                      self.lang_source,
                                                      self.lang_target,
                                                      lang_pivot)
            assert content_translated, 'Content translation is empty!'

            # store translation in cache
            with open(content_translations_file, 'w') as ct:
                ct.writelines(f'{translation}\n' for translation in content_translated)

        else:
            # get translations from cache
            logging.info(f"Get content translation from cache: {content_translations_file}")
            with open(content_translations_file) as ct:
                content_translated = [translation for translation in ct.readlines()]

        # Align
        aligner = Aligner(alignment_model=alignment_model,
                          tokenizer_src=self.tokenizer_src,
                          tokenizer_tgt=self.tokenizer_tgt)
        content_alignments = aligner.align(content,
                                           self.lang_source,
                                           content_translated,
                                           self.lang_target,
                                           self.alignment_type,
                                           self.output_dir)

        assert content_alignments, 'Alignment is empty!'

        # Add original sentence, the corresponding translations and the alignments
        content_translations_alignments = defaultdict()
        for sentence, sentence_translated, alignment in zip(content,
                                                            content_translated,
                                                            content_alignments):
            content_translations_alignments[sentence] = {'source': sentence,
                                                         'translation': sentence_translated,
                                                         'alignment': alignment}
        with open(content_translations_alignments_file, 'w') as fn:
            json.dump(content_translations_alignments, fn)

        return content_translations_alignments

    # Parse the SQUAD file and replace the questions, context and answers field with their translations
    # using the content_translations_alignments

    # For the answer translated the following two-steps logic is applied:
    # 1) Translate the answer and find them in the context translated
    #   1.1) searching around the answer start provided by the alignment
    #   1.2) if not the previous, searching from the beginning of the context translated
    #
    # 2) If the previous two steps fail, optionally extract the answer from the context translated
    # using the answer start and answer end provided by the alignment
    def translate_squad(self, sample_size, overwrite_cached_data, translation_engine, lang_pivot, alignment_model,
                        postprocess_answer):
        dataset = self.load_dataset(self.squad_file, sample_size=sample_size)

        content_translations_alignments = self.translate_align_content(dataset, overwrite_cached_data,
                                                                       translation_engine, lang_pivot, alignment_model)

        answer_retriever = AnswerRetriever(answer_retrieval=self.answer_retrieval,
                                           tokenizer_src=self.tokenizer_src,
                                           tokenizer_tgt=self.tokenizer_tgt)

        for data in tqdm(dataset['data']):
            # title = data['title']
            # data['title'] = content_translations_alignments[title]['translation']
            for paragraphs in data['paragraphs']:
                context = paragraphs['context']

                context_sentences = [s for s in
                                     self.sent_tokenizer.tokenize_sentences(utils.remove_line_breaks(context))]

                context_translated = ' '.join(content_translations_alignments[s]['translation']
                                              for s in context_sentences)
                context_alignment_tok = utils.compute_context_alignment(
                    [content_translations_alignments[s]['alignment']
                     for s in context_sentences])

                # Translate context and replace its value back in the paragraphs
                paragraphs['context'] = context_translated
                for qa in paragraphs['qas']:
                    question = qa['question']
                    question_translated = content_translations_alignments[question]['translation']
                    qa['question'] = question_translated

                    # Translate answers and plausible answers for SQUAD v2.0
                    if dataset['version'] == 'v2.0':
                        if not qa['is_impossible']:
                            for answer in qa['answers']:
                                answer_translated = content_translations_alignments[answer['text']]['translation']
                                answer_translated, answer_translated_start = \
                                    answer_retriever.retrieve_answer(answer,
                                                                     answer_translated,
                                                                     context,
                                                                     context_translated,
                                                                     context_alignment_tok,
                                                                     postprocess=postprocess_answer)

                                answer['text'] = answer_translated
                                answer['answer_start'] = answer_translated_start

                        else:
                            for plausible_answer in qa['plausible_answers']:
                                plausible_answer_translated = \
                                    content_translations_alignments[plausible_answer['text']]['translation']
                                answer_translated, answer_translated_start = \
                                    answer_retriever.retrieve_answer(plausible_answer,
                                                                     plausible_answer_translated,
                                                                     context,
                                                                     context_translated,
                                                                     context_alignment_tok,
                                                                     postprocess=postprocess_answer)
                                plausible_answer['text'] = answer_translated
                                plausible_answer['answer_start'] = answer_translated_start

                    # Translate answers for SQUAD v1.1
                    else:
                        for answer in qa['answers']:
                            answer_translated = content_translations_alignments[answer['text']]['translation']
                            answer_translated, answer_translated_start = \
                                answer_retriever.retrieve_answer(answer,
                                                                 answer_translated,
                                                                 context,
                                                                 context_translated,
                                                                 context_alignment_tok,
                                                                 postprocess=postprocess_answer)
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
                    if dataset['version'] == 'v2.0':
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
        translated_file = os.path.join(self.output_dir,
                                       os.path.basename(self.squad_file).replace(
                                           f'.json',
                                           f'-answer-retrieved-{self.answer_retrieval}-{self.lang_target}.json'))

        with open(translated_file, 'w') as fn:
            json.dump(content_cleaned, fn)

        # Count correct answers and plausible answers for SQUAD v2.0
        if dataset['version'] == 'v2.0':
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
    parser.add_argument('--answer_retrieval', type=str, choices=['span_plus_alignment', 'alignment', 'span'],
                        default='all',
                        help='Specify how translated answers are retrieved')
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
    parser.add_argument('--no_cuda', action='store_true', help='Do not use CUDA')
    parser.add_argument('--postprocess_answer', action='store_true',
                        help='Post-process retrieved answer with heuristics')
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

    if not args.no_cuda:
        DEVICE = torch.cuda.current_device()
    else:
        DEVICE = 'cpu'
    random.seed(args.seed)

    # Create output directory if doesn't exist already
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    tokenizer_src = Tokenizer(lang=args.lang_source)
    tokenizer_tgt = Tokenizer(lang=args.lang_target)

    squadtranslator = SquadTranslator(args.squad_file,
                                      args.lang_source,
                                      args.lang_target,
                                      args.output_dir,
                                      args.alignment_type,
                                      args.answer_retrieval,
                                      args.batch_size,
                                      tokenizer_src,
                                      tokenizer_tgt)

    logging.info(f'Translate the SQUAD dataset: {args.squad_file}')
    squadtranslator.translate_squad(args.sample_size, args.overwrite_cached_data, args.translation_engine,
                                    args.lang_pivot, args.alignment_model, args.postprocess_answer)

    end = time.time()
    logging.info(f'Total execution time: {round(end - start)} s')
