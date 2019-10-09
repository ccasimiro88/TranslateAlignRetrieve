from sacremoses import MosesTokenizer, MosesDetokenizer, MosesTruecaser, MosesDetruecaser, MosesPunctNormalizer
import subword_nmt.apply_bpe as apply_bpe
import re


# Instantiate processing objects
lang_src = 'en'
lang_tgt = 'es'

# For source pre-processing
punct_normalizer = MosesPunctNormalizer()
tokenizer_src = MosesTokenizer(lang=lang_src)

truecase_model_src = '/home/casimiro/projects/hutoma/nmt_en-es_general//data/en2es/preprocess/truecase-model.en'
truecaser_src = MosesTruecaser(load_from=truecase_model_src)

bpe_codes_joint = '/home/casimiro/projects/hutoma/nmt_en-es_general//data/en2es/preprocess/joint_bpe'
bpe_vocabulary_src = '/home/casimiro/projects/hutoma/nmt_en-es_general//data/en2es/preprocess/vocab.en'
bpe_pattern = r'(@@ )|(@@ ?$)'
bpe_vocab_src = apply_bpe.read_vocabulary(open(bpe_vocabulary_src), threshold=50)
bpe_segmenter_src = apply_bpe.BPE(codes=open(bpe_codes_joint), vocab=bpe_vocab_src)

# For predictions postprocessing
detokenizer_pred = MosesDetokenizer(lang=lang_tgt)
detruecaser_pred = MosesDetruecaser()
truecaser_model = {'en': truecaser_src}


# The pre-processing for source text consists of:
# 1)normalization of punctuation, 2) tokenization, 3) truecasing, 4)BPE segmentation
def preprocess_src(text):
    text_prepro = punct_normalizer.normalize(text)
    text_prepro = tokenizer_src.tokenize(text_prepro, return_str=True, escape=False)
    text_prepro = truecaser_src.truecase(text_prepro)
    text_prepro = bpe_segmenter_src.segment_tokens(text_prepro)
    return ' '.join(text_prepro)


# The post-processing for predictions consists of:
# 1) restore the original segmentation before bpe, 2) detruecase, 4) detokenize
def postprocess_pred(text):
    text_postpro = re.sub(bpe_pattern, r'', text)
    text_postpro = detruecaser_pred.detruecase(text_postpro)
    text_postpro = detokenizer_pred.detokenize(text_postpro, return_str=True)
    return text_postpro


def test_apply_bpe(toks):
    print(bpe_segmenter_src.segment_tokens(toks))


tokens_en=['&quot;', 'Norman', '&quot;', 'comes', 'from', '&quot;', 'Norseman', '&quot;']
tokens_es=['&quot;', 'comida', '&quot;']
test_apply_bpe(tokens_en)
