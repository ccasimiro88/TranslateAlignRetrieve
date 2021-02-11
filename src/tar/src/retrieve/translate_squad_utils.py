import requests
import subprocess
import json
import os
import tempfile
from sacremoses import MosesTokenizer, MosesDetokenizer
from collections import defaultdict
from nltk import sent_tokenize
import sentence_splitter

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# PROCESSING TEXT
MAX_NUM_TOKENS = 10
SPLIT_DELIMITER = ';'
LANGUAGE_ISO_MAP = {'en': 'english', 'es': 'spanish'}


def tokenize(text, lang, return_str=True):
    return MosesTokenizer(lang=lang).tokenize(text, return_str=return_str, escape=False)


def de_tokenize(text, lang):
    if not isinstance(text, list):
        text = text.split()
    return MosesDetokenizer(lang=lang).detokenize(text, return_str=True)


# Chunk sentences longer than a maximum number of words/tokens based on a delimiter character.
# This option is used only for very long sentences to avoid shorter translation than the
# original source length.
# Note that the delimiter can't be a trailing character
def split_sentences(text, lang, delimiter=SPLIT_DELIMITER, max_size=MAX_NUM_TOKENS, tokenized=True):
    text_len = len(tokenize(text, lang, return_str=True).split()) if tokenized else len(text.split())
    if text_len >= max_size:
        delimiter_match = delimiter + ' '
        text_chunks = [chunk.strip() for chunk in text.split(delimiter_match) if chunk]
        # Add the delimiter lost during chunking
        text_chunks = [chunk + delimiter for chunk in text_chunks[:-1]] + [text_chunks[-1]]
        return text_chunks
    return [text]


def tokenize_sentences(text, lang):
    sentences = [chunk
                 for sentence in sentence_splitter.SentenceSplitter(language=lang).split(text)
                 for chunk in split_sentences(sentence, lang)]
    return sentences


# SQUAD paragraphs contains line breaks that we have to remove
def remove_line_breaks(text):
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    return text


# Remove trailing punctuation from the answers retrieved from alignment
def remove_extra_punct(source, translation):
    periods_commas = '.,;:'
    brackets = [['(', ')'], ['[', ']'], ['{', '}']]
    exclamation = '¡!'
    quotation = '"'
    try:
        if len(translation) != 1:
            # Remove extra periods or commas
            if source[-1] in periods_commas and translation[-1] in periods_commas:
                translation = translation
            else:
                if source[-1] in periods_commas and translation[-1] not in periods_commas:
                    translation = translation + source[-1]
                elif source[-1] not in periods_commas and translation[-1] in periods_commas:
                    translation = translation.strip(translation[-1])
            # remove brackets
            if translation[-1] in [b[1] for b in brackets] and any(
                    c for c in translation if c in [b[0] for b in brackets]):
                translation = translation
            else:
                for bracket in brackets:
                    if translation[-1] in bracket[1] and translation[0] not in bracket[0]:
                        translation = translation[:-1]
                    elif translation[-1] not in bracket[1] and translation[0] in bracket[0]:
                        translation = translation[1:]
            # Complete exclamation mark
            if translation[-1] in exclamation and translation[0] in exclamation:
                translation = translation
            else:
                if translation[-1] in exclamation and translation[0] not in exclamation:
                    translation = exclamation[0] + translation
                elif translation[-1] not in exclamation and translation[0] in exclamation:
                    translation = translation + exclamation[1]
            # Complete quotation
            if translation[-1] in quotation and translation[0] in quotation:
                translation = translation
            else:
                if translation[-1] in quotation and translation[0] not in quotation:
                    translation = quotation + translation
                elif translation[-1] not in quotation and translation[0] in quotation:
                    translation = translation + quotation
            return translation
        else:
            return translation
    except IndexError:
        return translation


# Keep the first part when the answer translation come across
# two sentences or when there are extra commas with words
def remove_extra_text(source, translation, lang='es'):
    translation = sentence_splitter.SentenceSplitter(language=lang).split(translation)[0]
    if ', ' in translation and ', ' not in source:
        translation = translation.split(', ')[0]
    return translation


# This function post-process the retrieved answer translation
# Note that the logic is applied on language-specific punctuation
# that means might not be generalize to every language
# TODO: generalize this logic to all the languages
def post_process_answers_translated(source, translation):
    # Keep the original answer when it is translation-invariant
    # like dates or proper names
    if len(source) > 1 and source in translation:
        translation = source
    # Post-process the answer translated
    else:
        translation = translation.strip()
        translation = remove_extra_text(source, translation)
        translation = remove_extra_punct(source, translation)
    return translation


# ALIGNMENT INDEX MANIPULATION
# Shift index in an alignment by a given amount in a give direction
def shift_value_index_alignment(value_index, alignment, direction='right'):
    # Remove duplicates while keeping the order of occurrence
    value_indexes = sorted(set(alignment.values()),
                           key=list(alignment.values()).index)
    if direction == 'right':
        next_value_index = value_indexes.index(value_index) + 1
        if next_value_index != len(value_indexes):
            return value_indexes[next_value_index]
        else:
            return -1
    else:
        next_value_index = value_indexes.index(value_index) - 1
        if next_value_index != 0:
            return value_indexes[next_value_index]
        else:
            return 0


# Get the closest left or right index in a list given a certain number
def get_left_right_close_index(indexes, number, type):
    if indexes:
        indexes_list = list(indexes)
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


# Compute alignment between character indexes and token indexes, and conversely
def tok2char_map(text_raw, text_tok):
    # First, compute the token to white-spaced token indexes map (many-to-one map)
    # tok --> ws_tok
    tok2ws_tok = dict()
    ws_tokens = text_raw.split()
    idx_wst = 0
    merge_tok = ''
    for idx_t, t in enumerate(text_tok.split()):
        merge_tok += t
        tok2ws_tok[idx_t] = idx_wst
        if merge_tok == ws_tokens[idx_wst]:
            idx_wst += 1
            merge_tok = ''

    # Second, compute white-spaced token to character indexes map (one-to-one map):
    # ws_tok  --> char
    ws_tok2char = dict()
    for ws_tok_idx, ws_tok in enumerate(text_raw.split()):
        if ws_tok_idx == 0:
            char_idx = 0
            ws_tok2char[ws_tok_idx] = char_idx
        elif ws_tok_idx > 0:
            char_idx = len(' '.join(text_raw.split()[:ws_tok_idx])) + 1
            ws_tok2char[ws_tok_idx] = char_idx

    # Finally, compute the token to character map (one-to-one)
    tok2char = {tok_idx: ws_tok2char[tok2ws_tok[tok_idx]]
                for tok_idx, _ in enumerate(text_tok.split())}

    return tok2char


# Convert a token-level alignment into a char-level alignment
# Can be white-spaced tokens or normal tokens, the important thing is the one-to-one mapping
def get_src2tran_alignment_char(alignment, source, translation):
    source_tok = tokenize(source, 'en')
    translation_tok = tokenize(translation, 'es')
    src_tok2char = tok2char_map(source, source_tok)
    tran_tok2char = tok2char_map(translation, translation_tok)

    # Get token index to char index translation map for both source and target
    src2tran_alignment_char = defaultdict(list)
    # Prevent
    try:
        for src_tran in alignment.split():
            src_tok_idx = int(src_tran.split('-')[0])
            tran_tok_idx = int(src_tran.split('-')[1])
            src_char_idx = src_tok2char[src_tok_idx]
            tran_char_idx = tran_tok2char[tran_tok_idx]
            src2tran_alignment_char[src_char_idx].append(tran_char_idx)
    except KeyError:
        pass
    # Define a one-to-one mapping left-oriented by keeping the minimum key value
    src2tran_alignment_char_min_tran_index = {k: min(v) for k, v in src2tran_alignment_char.items()}

    return src2tran_alignment_char_min_tran_index


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


# TRANSLATING
# Translate text using the OpenNMT-py script
PUNCTUATION = ['.', ',', '?', '!', '¿', '¡', ')', '(', ']', '[']


# Remove extra punctuation when the translations come from a very short text
# Handle exceptions in case the translation in just one word length
def post_process_translation(source, translation, punctuation=PUNCTUATION):
    try:
        # Avoid translations where the same token is repeated
        if set(translation.split()) == translation.split()[0]:
            translation = translation[0]
        if source[0].isupper():
            translation = translation[0].upper() + translation[1:]
        if source[0].islower() and len(translation) > 1:
            translation = translation[0].lower() + translation[1:]
        if source[-1] == '.':
            if translation[-1] in punctuation:
                translation = translation[:-1] + '.'
            else:
                translation += '.'
        if source[-1] == ',':
            if translation[-1] in punctuation:
                translation = translation[:-1] + ','
            else:
                translation += ','
        if translation[0] in punctuation and len(translation) > 1:
            translation = translation[1:]
        return translation
    except IndexError:
        return translation
