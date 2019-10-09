import jsonlines

filename = '/home/casimiro/projects/hutoma/qa_bert_fine-tuning/corpora/' \
           'natural_questions/v1.0/sample/nq-dev-sample.1.jsonl'


def translate(tokens):
    return []


# Read lines from jsonl file
with jsonlines.open(filename) as lines:

    # Extract question and documents tokens
    for line in lines:
        question_tokens = line['question_tokens']
        document_tokens = [content['token'] for content in line['document_tokens']]

        # Translate question and documents tokens and replace them in the original document
        line['question_tokens'] = translate(question_tokens)
        document_tokens_translated = translate(document_tokens)


# # Extract all question and documents tokens at once
# questions_tokens = [line['question_tokens'] for line in lines]
# documents_tokens = [[document_tokens['token'] for document_tokens in line['document_tokens']] for line in lines]
