# This script clean a parallel corpora from sentence that
# are not in the correct source/target language and
# then splits it up into train/dev/test datasets
import fasttext
import argparse
import os
import random
import time
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

FASTTEXT_LANG_DETECT_MODEL = SCRIPT_DIR + '/data/fastText/lid.176.bin'
langdetect = fasttext.load_model(FASTTEXT_LANG_DETECT_MODEL)


def check_correct_target_language(text, target_language):
    prediction = langdetect.predict(text)
    label_language = prediction[0][0]
    return label_language.endswith(target_language)


def create_datasets(source_file, target_file, source_lang, target_lang, output_dir, test_size, valid_size):
    with open(source_file) as sf, open(target_file) as tf:
        source_target_lines = [(s.strip(), t.strip()) for (s, t) in tqdm(zip(sf.readlines(), tf.readlines()))]

    # Remove pairs duplicates
    source_target_lines = set(source_target_lines)

    # Remove pairs containing source or target duplicates
    source_seen = set()
    target_seen = set()
    source_target_lines_clean = set()
    for st in tqdm(source_target_lines):
        if st[0] not in source_seen and st[1] not in target_seen:
            source_seen.add(st[0])
            target_seen.add(st[1])
            source_target_lines_clean.add(st)

    # Remove pairs with wrongly aligned (uncorrect translations)
    source_target_lines_clean = [st for st in tqdm(source_target_lines_clean)
                                 if check_correct_target_language(st[0], source_lang)
                                 and check_correct_target_language(st[1], target_lang)
                                 and st[0] != st[1]]

    # Shuffle and split
    random.seed(10)
    random.shuffle(source_target_lines_clean)
    source_target_lines_test = source_target_lines_clean[:test_size+1]
    source_target_lines_valid = source_target_lines_clean[test_size+1:(valid_size+test_size)+2]
    source_target_lines_train = source_target_lines_clean[(valid_size+test_size)+2:]

    source_test = os.path.join(output_dir, 'test.{}'.format(source_lang))
    target_test = os.path.join(output_dir, 'test.{}'.format(target_lang))
    with open(source_test, 'w', encoding='utf8') as sf, open(target_test, 'w') as tf:
        sf.writelines('\n'.join(st[0] for st in tqdm(source_target_lines_test)))
        tf.writelines('\n'.join(st[1] for st in tqdm(source_target_lines_test)))

    source_valid = os.path.join(output_dir, 'valid.{}'.format(source_lang))
    target_valid = os.path.join(output_dir, 'valid.{}'.format(target_lang))
    with open(source_valid, 'w', encoding='utf8') as sf, open(target_valid, 'w') as tf:
        sf.writelines('\n'.join(st[0] for st in tqdm(source_target_lines_valid)))
        tf.writelines('\n'.join(st[1] for st in tqdm(source_target_lines_valid)))

    source_train = os.path.join(output_dir, 'train.{}'.format(source_lang))
    target_train = os.path.join(output_dir, 'train.{}'.format(target_lang))
    with open(source_train, 'w', encoding='utf8') as sf, open(target_train, 'w') as tf:
        sf.writelines('\n'.join(st[0] for st in tqdm(source_target_lines_train)))
        tf.writelines('\n'.join(st[1] for st in tqdm(source_target_lines_train)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type=str, help='Source file')
    parser.add_argument('--target_file', type=str, help='Target file')
    parser.add_argument('--source_lang', type=str, help='Source lang')
    parser.add_argument('--target_lang', type=str, help='Target lang')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--test_size', type=int, help='Test dataset size')
    parser.add_argument('--valid_size', type=int, help='Valid dataset size')
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    start = time.time()
    create_datasets(args.source_file, args.target_file, args.source_lang, args.target_lang,
                    args.output_dir,
                    args.test_size, args.valid_size)
    end = time.time()
    print('Total time: {} s'.format(end-start))

