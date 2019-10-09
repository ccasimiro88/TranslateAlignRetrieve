import json
from langdetect import detect, DetectorFactory
import argparse

DetectorFactory.seed = 0


def check_correct_target_language(text, target_language):
    try:
        translation_language = detect(text)
        return bool(translation_language == target_language)
    except:
        return True


def compute_percentage_correct_translations(squad_file, target_language):
    with open(squad_file) as fn:
        content = json.load(fn)

    incorrect_translations = {'questions': [], 'context': []}
    questions_count = 0
    for data in content['data']:
        for paragraphs in data['paragraphs']:
            for qas in paragraphs['qas']:
                question = qas['question']
                questions_count += 1
                # Language detection is not enough accurate, that's why I look for the interrogation mark at the
                # beginning of the question (this works only for spanish)
                if not check_correct_target_language(question, target_language) and not question.startswith('¿'):
                    incorrect_translations['questions'].append(question)

    percentage_questions = round((questions_count - len(incorrect_translations['questions']))/questions_count, 2)*100
    print('Percentage of correct translated questions: {}%'.format(percentage_questions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-squad_file', type=str)
    parser.add_argument('-target_lang', type=str)

    args = parser.parse_args()

    compute_percentage_correct_translations(args.squad_file, args.target_lang)
