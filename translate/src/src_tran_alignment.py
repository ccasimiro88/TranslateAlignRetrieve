from collections import defaultdict
from sacremoses import MosesTokenizer

mt = MosesTokenizer(lang='en')


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
        source_tok = mt.tokenize(source, return_str=True, escape=False)
        tran_tok = mt.tokenize(translation, return_str=True, escape=False)

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


if __name__ == '__main__':
    def test():
        source = 'The Duchy of Normandy, which began in 911 as a fiefdom, was established by the treaty of ' \
                 'Saint-Clair-sur-Epte between King Charles III of West Francia and the famed Viking ruler Rollo, ' \
                 'and was situated in the former Frankish kingdom of Neustria.'
        translation = 'El Ducado de Normandía, que comenzó en 911 como feudo, fue establecido por el tratado de ' \
                      'Saint-Clair-sur-Epte entre el rey Carlos III de Francia Occidental y el famoso gobernante ' \
                      'vikingo Rollo, y estaba situado en el antiguo reino franco.'
        alignment_raw = '0-0 1-1 2-2 3-3 4-4 5-5 6-6 7-7 8-8 10-9 11-10 12-11 13-12 14-13 15-14 16-15 17-16 18-17 ' \
                        '19-19 20-20 21-21 22-22 23-23 24-24 25-25 26-26 27-27 28-29 30-30 31-31 32-32 33-33 ' \
                        '34-34 35-35 37-36 38-37 40-38'
        alignment_tok = '0-0 1-1 2-2 3-3 4-4 5-5 6-6 7-7 8-8 9-9 11-10 12-11 13-12 14-13 15-14 16-15 17-16 18-17 ' \
                        '19-18 20-19 21-21 22-22 23-23 24-24 25-25 26-26 27-27 28-28 29-29 31-30 30-31 32-32 33-33 ' \
                        '34-34 35-35 36-36 37-37 38-38 39-39 41-40 43-41 44-42'

        src_tran_alignment_tokens = get_src_tran_char_alignment(alignment_tok, source, translation, level='tokens')
        src_tran_alignment_raw = get_src_tran_char_alignment(alignment_raw, source, translation, level='raw')
        return src_tran_alignment_raw, src_tran_alignment_tokens

    test()

