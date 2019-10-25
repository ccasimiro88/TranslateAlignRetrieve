# This script contains utils function for the translate_squad.py
import requests
import subprocess
import json
import os
import tempfile
from sacremoses import MosesTokenizer, MosesDetokenizer
import fasttext
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# PROCESSING TEXT
tokenizer_en = MosesTokenizer(lang='en')
detokenizer_en = MosesDetokenizer(lang='en')
tokenizer_es = MosesTokenizer(lang='es')
detokenizer_es = MosesDetokenizer(lang='es')

# Check if target language for translations using FastText model
FASTTEXT_LANG_DETECT_MODEL = SCRIPT_DIR + '/../data/fastText/lid.176.bin'
langdetect = fasttext.load_model(FASTTEXT_LANG_DETECT_MODEL)


def check_correct_target_language(text, target_language):
    prediction = langdetect.predict(text)
    label_language = prediction[0][0]
    return label_language.endswith(target_language)


def tokenize(text, lang):
    if lang == 'en':
        text_tok = tokenizer_en.tokenize(text, return_str=True, escape=False)
        return text_tok
    elif lang == 'es':
        text_tok = tokenizer_es.tokenize(text, return_str=True, escape=False)
        return text_tok


def de_tokenize(text, lang):
    if not isinstance(text, list):
        text = text.split()

    if lang == 'en':
        text_detok = detokenizer_en.detokenize(text, return_str=True)
        return text_detok
    elif lang == 'es':
        text_detok = detokenizer_es.detokenize(text, return_str=True)
        return text_detok


def post_process_translation(text):
    text = text.strip()

    # Add leading question mark when is missing (for Spanish)
    if text.endswith('?'):
        text = '¿' + text
    return text


# SQUAD paragraphs contains line breaks that we have to remove
def remove_line_breaks(text):
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    return text


# Remove trailing punctuation from the answers retrieved from alignment
def remove_trailing_punct(text):
    if text and text[-1] in [',', '.']:
        text = text[:-1]
        return text
    else:
        return text


# ALIGNMENT INDEX MANIPULATION
# Shift index by a given amount in a give direction
def shift_index(index, shift, direction='left'):
    if direction == 'left' and index - shift > 0:
        return index - shift
    else:
        return index


# Get the closest left or right index in a list given a certain number
def get_left_right_close_index(indexes_list, number, type):
    if isinstance(indexes_list, list) and indexes_list:
        # The >= condition is necessary to return exactly the same number if is in the list
        if type == 'left':
            if number > min(indexes_list):
                left_indexes = [idx for idx in indexes_list if (number - idx) >= 0]
                left_close_index = min(left_indexes, key=lambda x: abs(number - x))
                return left_close_index
            else:
                return min(indexes_list)

        elif type == 'right':
            if number < max(indexes_list):
                right_indexes = [idx for idx in indexes_list if (number - idx) <= 0]
                right_close_index = min(right_indexes, key=lambda x: abs(number - x))
                return right_close_index
            else:
                return max(indexes_list)
    else:
        return number


# Compute alignment between token indexes and white-spaced token indexes
def tok_wstok_align(raw, tok):
    tok2ws_tok = defaultdict(list)
    ws_tok2tok = defaultdict(list)
    ws_tokens = raw.split()
    idx_wst = 0
    merge_tok = ''
    for idx_t, t in enumerate(tok.split()):
        # import pdb; pdb.set_trace()
        merge_tok += t
        tok2ws_tok[idx_t] = idx_wst
        if merge_tok == ws_tokens[idx_wst]:
            idx_wst += 1
            merge_tok = ''

    for idx_t, idx_wst in tok2ws_tok.items():
        ws_tok2tok[idx_wst].append(idx_t)
    return dict(tok2ws_tok), dict(ws_tok2tok)


# Convert a token-level alignment into a char-level alignment
def get_src_tran_char_alignment(alignment, source, translation, level='raw'):
    # First, extract map between token indexes and char indexes for both source and target
    # The raw level assumes the text is not tokenized
    if level == 'raw':
        src_token_index = [int(src_tgt_idx.split('-')[0])
                           for src_tgt_idx in alignment.split()]
        tran_token_index = [int(src_tran_idx.split('-')[1])
                            for src_tran_idx in alignment.split()]

    # The tokens level works on tokenized text. So, we first map the white-spaced token indexes to the token indexes
    # and then compute the char indexes
    elif level == 'tokens':
        source_tok = tokenize(source, 'en')
        tran_tok = tokenize(translation, 'es')

        # Map from white-spaced token indexes to token indexes
        source_tok2ws_tok, _ = tok_wstok_align(source, source_tok)
        tran_tok2ws_tok, _ = tok_wstok_align(translation, tran_tok)

        src_token_index = [source_tok2ws_tok[int(src_tgt_idx.split('-')[0])]
                           for src_tgt_idx in alignment.split()]
        tran_token_index = [tran_tok2ws_tok[int(src_tran_idx.split('-')[1])]
                            for src_tran_idx in alignment.split()]

    src_token_index2char_index = {}
    for src_idx in src_token_index:
        if src_idx == 0:
            src_token_index2char_index[src_idx] = 0
        elif src_idx > 0:
            src_token_index2char_index[src_idx] = len(' '.join(source.split()[:src_idx])) + 1

    tran_token_index2char_index = {}
    for tran_idx in tran_token_index:
        if tran_idx == 0:
            tran_token_index2char_index[tran_idx] = 0
        elif tran_idx > 0:
            tran_token_index2char_index[tran_idx] = len(' '.join(translation.split()[:tran_idx])) + 1

    # Then, use the previous maps to create the final src-tgt char alignment
    src_tran_char_alignment = {src_token_index2char_index[src_tok_idx]: tran_token_index2char_index[tran_tok_idx]
                               for src_tok_idx, tran_tok_idx in zip(src_token_index, tran_token_index)}
    return src_tran_char_alignment


# Convert a set of sentence alignments into one document alignment
def compute_context_alignment(sentence_alignments):
    if isinstance(sentence_alignments, list) and len(sentence_alignments) > 1:
        def get_max_src_tgt_token_index(sentence_alignment):
            src_token_index = [int(src_tgt_idx.split('-')[0]) for src_tgt_idx in sentence_alignment.split()]
            tran_token_index = [int(src_tran_idx.split('-')[1]) for src_tran_idx in sentence_alignment.split()]
            shift_source = max(src_token_index) + 1
            shift_translation = max(tran_token_index) + 1
            return shift_source, shift_translation

        def shift_alignment(alignment, shift_source, shift_translation):
            shifted_alignment = ' '.join(['{}-{}'.format((int(src_tgt_idx.split('-')[0]) + shift_source),
                                                         (int(src_tgt_idx.split('-')[1]) + shift_translation))
                                          for src_tgt_idx in alignment.split()]).strip()
            return shifted_alignment

        # Add shift for source and target token index only to the alignments for the second-to-last sentences.
        # Also, the shift value increase while looping over the sentence alignments
        context_alignment = ''
        for idx_alignment, sent_alignment in enumerate(sentence_alignments):
            if idx_alignment == 0:
                shift_src, shift_tran = 0, 0
                context_alignment += shift_alignment(sent_alignment, shift_src, shift_tran)
            elif idx_alignment > 0:
                shift_src, shift_tran = get_max_src_tgt_token_index(context_alignment)
                context_alignment += ' ' + shift_alignment(sent_alignment, shift_src, shift_tran)
        context_alignment = context_alignment.strip()
    else:
        # Convert to string
        context_alignment = ''.join(sentence_alignments).strip()
    return context_alignment


# ANSWER EXTRACTION FROM CONTEXT
# This function compute the span of the answer translated based on
# how many words are present in the original answer
def length_based_answer_extraction(start_index, next_start_index, original_answer, alignment_indexes):
    def get_indexes_in_between_range(start_index, next_start_index):
        return len(set([i for i in alignment_indexes if i >= start_index]).intersection(
            set([i for i in alignment_indexes if i < next_start_index])))

    if isinstance(alignment_indexes, list) and alignment_indexes:
        answer_word_length = len(original_answer.split())
        span_word_length = get_indexes_in_between_range(start_index, next_start_index)
        shift = answer_word_length - span_word_length
        # Shift to the index to the next items in the alignment translation indexes
        try:
            # Enlarge or reduce the span on the ordered indexes
            start_index = min(start_index, next_start_index)
            next_start_index = max(start_index, next_start_index)
            next_start_span = alignment_indexes[alignment_indexes.index(next_start_index) + shift]
            return next_start_span
        except IndexError:
            return next_start_index
    else:
        return next_start_index


# This function extract the answer translated from the context translate only using context alignment.
# A series of heuristic are applied in order to extract the answer
def extract_answer_translated_from_alignment(answer_text, answer_start, context_translated, context_alignment):
    # Get the corresponding start and end char of the answer_translated in the
    # context translated and retrieve answer translated.

    def extract_answer_from_start_next_start_indexes(start, next_start, context):
        # the answer_translated is a span from the min to max char index (account for the inversion
        # of words during translation)
        start = min(start, next_start)
        next_start = max(start, next_start)
        ans = context[start: next_start].strip()
        ans = remove_trailing_punct(ans)
        return ans, start

    answer_next_start = answer_start + len(answer_text) + 1
    left_close_answer_start = get_left_right_close_index(list(context_alignment.keys()), answer_start, type='left')
    left_close_answer_translated_start = context_alignment[left_close_answer_start]

    # For the answer end I'll try to use the left-close index (if it is different than the
    # answer start) otherwise I use the right-close index
    # Answer next start left close
    left_close_answer_next_start = get_left_right_close_index(list(context_alignment.keys()),
                                                              answer_next_start,
                                                              type='left')

    if left_close_answer_next_start != left_close_answer_start:
        left_close_answer_translated_next_start = context_alignment[left_close_answer_next_start]

        # Check it the retrieved start and next start indexes contain the same number of words in the answer.
        # If not, shift the next start index to the right
        left_close_answer_translated_next_start = length_based_answer_extraction(left_close_answer_translated_start,
                                                                                 left_close_answer_translated_next_start,
                                                                                 answer_text,
                                                                                 list(context_alignment.values()))

        # Extract answer translated from context translated with answer_start and answer_next_start
        answer_translated, answer_translated_start = \
            extract_answer_from_start_next_start_indexes(left_close_answer_translated_start,
                                                         left_close_answer_translated_next_start,
                                                         context_translated)

    # Answer next start right-close
    elif left_close_answer_next_start == left_close_answer_start:
        # Check if there is a right-close value
        if answer_next_start <= max(context_alignment.keys()):
            right_close_answer_next_start = get_left_right_close_index(list(context_alignment.keys()),
                                                                       answer_next_start,
                                                                       type='right')
            right_close_answer_translated_next_start = context_alignment[right_close_answer_next_start]

            # Check it the retrieved start and next start indexes contain the same number of words in the answer.
            # If not, shift the next start index to the right
            right_close_answer_translated_next_start = length_based_answer_extraction(
                left_close_answer_translated_start,
                right_close_answer_translated_next_start,
                answer_text,
                list(context_alignment.values()))

            # Extract answer translated from context translated with answer_start and answer_next_start
            answer_translated, answer_translated_start = \
                extract_answer_from_start_next_start_indexes(left_close_answer_translated_start,
                                                             right_close_answer_translated_next_start,
                                                             context_translated)

        # No retrieved answer translated and answer translated index
        else:
            answer_translated = ''
            answer_translated_start = -1
    else:
        answer_translated = ''
        answer_translated_start = -1

    # TODO: create a document-dependent filename to store the answer retrieved from alignment
    # Write answers extracted from alignment to stdout for later check the accuracy
    print('ORIGINAL: {} | {}'.format(answer_text, answer_start))
    print('FROM ALIGNMENT: {} | {}\n'.format(answer_translated, answer_translated_start))

    return answer_translated, answer_translated_start


# This function extract the answer from a given context
def extract_answer_translated(answer, answer_translated,
                              context_translated, context_alignment,
                              retrieve_answers_from_alignment):
    # 1.1)
    # Match the answer_translated in the context_translated starting from the left-close char index
    # of the answer_start in the context_alignment. When the answer_start is present in the
    # context_alignment the closest index is exactly the answer_start index
    answer_text = answer['text']
    answer_start = answer['answer_start']
    left_close_answer_start = get_left_right_close_index(list(context_alignment.keys()),
                                                         answer_start,
                                                         type='left')
    try:
        left_close_answer_translated_start = context_alignment[left_close_answer_start]
    except KeyError:
        answer_translated, answer_translated_start = '', -1
    # Find answer_start in the translated context by looking close to the answer_translated_char_start
    # I am shifting the index by an additional 20 chars to the left in the case
    # the alignment is not precise enough (20 chars is more or less 2/3 words)
    # TODO: use regex match in order to take into account word boundaries and avoid substring match
    # within the words
    if context_translated.lower().find(answer_translated.lower(),
                                       shift_index(left_close_answer_translated_start, 20)) != -1:

        answer_translated_start = context_translated.lower().find(answer_translated.lower(),
                                                                  shift_index(left_close_answer_translated_start, 20))
        answer_translated_end = answer_translated_start + len(answer_translated)
        answer_translated = context_translated[answer_translated_start: answer_translated_end + 1]

    # 1.2) Find the answer_translated in the context_translated from the beginning of the text
    elif context_translated.lower().find(answer_translated.lower(),
                                         left_close_answer_translated_start) == -1:
        if context_translated.lower().find(answer_translated.lower()) != -1:

            answer_translated_start = context_translated.lower().find(answer_translated.lower())
            answer_translated_end = answer_translated_start + len(answer_translated)
            answer_translated = context_translated[answer_translated_start: answer_translated_end + 1]

        # 2) Retrieve the answer from the context translated using the
        # answer start and answer end provided by the alignment
        else:
            if retrieve_answers_from_alignment:
                answer_translated, answer_translated_start = \
                    extract_answer_translated_from_alignment(answer_text, answer_start,
                                                             context_translated, context_alignment)
            else:
                answer_translated = ''
                answer_translated_start = -1

    # No answer translated found
    else:
        answer_translated = ''
        answer_translated_start = -1

    return answer_translated, answer_translated_start


# TRANSLATING
# Translate text via an OpenNMT-based web service
# TODO: Later, put these variables in a config file
ENTOES_TRANSLATION_SERVICE_URL = 'http://10.8.0.22:5100/translator/translate'
MODEL_ID = 100


def translate(text, service_url=ENTOES_TRANSLATION_SERVICE_URL):
    headers = {'Content-Type': 'application/json'}
    data = [{"src": "", "id": MODEL_ID}]

    def get_translation(text):
        data[0]['src'] = text
        response = requests.post(url=service_url, headers=headers, json=data)
        return json.loads(response.text)[0][0]['tgt']

    # Get translation
    translation = get_translation(text)

    return translation


# Translate via script
def translate_script(source_sentences, output_dir, batch_size):

    source_filename = os.path.join(output_dir, 'source_translate')
    with open(source_filename, 'w') as sf:
        sf.writelines('\n'.join(s for s in source_sentences))

    translation_filename = os.path.join(output_dir, 'target_translated')
    en2es_translate_cmd = SCRIPT_DIR + '/en2es_translate.sh {} {} {}'.format(source_filename,
                                                                             translation_filename,
                                                                             batch_size)
    subprocess.run(en2es_translate_cmd.split())

    with open(translation_filename) as tf:
        translated_sentences = [s.strip() for s in tf.readlines()]

    os.remove(source_filename)
    os.remove(translation_filename)

    return translated_sentences


# COMPUTE ALIGNMENT
# Compute alignment between source and target sentences
def compute_efolmal_sentence_alignment(source_sentences, source_lang,
                                       translated_sentences, target_lang,
                                       alignment_type, output_dir):

    source_filename = os.path.join(output_dir, 'source_align')
    with open(source_filename, 'w') as sf:
        sf.writelines('\n'.join(s for s in source_sentences))

    translation_filename = os.path.join(output_dir, 'target_align')
    with open(translation_filename, 'w') as tf:
        tf.writelines('\n'.join(s for s in translated_sentences))

    alignment_filename = os.path.join(output_dir, 'alignment')
    efolmal_cmd = SCRIPT_DIR + '/get_alignment_eflomal.sh {} {} {} {} {} {}'.format(source_filename,
                                                                                    source_lang,
                                                                                    translation_filename,
                                                                                    target_lang,
                                                                                    alignment_type,
                                                                                    alignment_filename)
    subprocess.run(efolmal_cmd.split())

    with open(alignment_filename) as af:
        alignments = [a.strip() for a in af.readlines()]

    os.remove(source_filename)
    os.remove(translation_filename)
    os.remove(alignment_filename)
    return alignments
