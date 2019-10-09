import jsonlines
import json

file2 = '/mnt/storage/paula/bert_nq/data/tiny-dev/nq-dev-sample.jsonl'
# file2 = '/mnt/storage/paula/datasets/natural_questions/v1.0/train/nq-train-00.jsonl'

# with jsonlines.open(file2, mode='r') as reader:
#     first = reader.read()

# tokens_text = []
# tokens_all = []
# for token in first['document_tokens']:
#   tokens_all.append(token['token'])
#   if not token['html_token']:
#     tokens_text.append(token['token'])


# print(' '.join(tokens_all))
# print(' '.join(tokens))


# html_tokens = []

from run_nq import create_example_from_jsonl
from os import listdir
from os.path import isfile, join

mypath = '/mnt/storage/paula/datasets/natural_questions/v1.0/train/'
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

i = 1
h = 0
m=0

examples = []
for file in onlyfiles:
  print(h)
  h += 1

  with open(file, mode='r') as reader:
      

      for line in reader.readlines():
        i+=1
#       print(i)
#       print("")
# # print(i)

        linej = json.loads(line)
        # print(linej['annotations'])
        example = create_example_from_jsonl(line) # this gets all the correct text without the html bits

        # if len(example['questions'])>1 or len(example['answers'])>1:
        # print(example['questions'])
        # print(example['answers'])
        # print("")
        #   print("")
        # i += 1


        squad_ex = {'title':example['name']}

        text = example['contexts']
        tokens = text.split(' ')
        new_tokens = []
        for token in tokens:
          if "[Paragraph=" in token:
            new_tokens.append("\n")
          elif token[0]=="[" and token[-1]=="]":
            continue
          else:
            new_tokens.append(token)
        context = ' '.join(new_tokens)

     
        paragraph = {'context': context}

        questions = [quest['input_text'] for quest in example['questions']]
        answers = [{'text': ans['span_text'], 'answer_start': ans['span_start']} for ans in example['answers'] if ans['input_text']=='short'] 

        qas = []

        if not answers==[]:
          m +=1
          for j in range(len(questions)):
            qas.append({'question':questions[j], 'answers':[answers[j]], 'id': i})
            # i += 1

          paragraph['qas']=qas
          squad_ex['paragraphs'] = [paragraph]
          examples.append(squad_ex)

print(i)
print(m)
squad = {'version': 'lixe', 'data': examples}

with open('squad_dev_shortans.json', 'w+') as t:
  json.dump(squad, t)

# with open('squad_dev_shortans.json', 'r') as t:
#   data = json.load(t)

# print(data['data'][44])
# print(data['data'][44]['paragraphs'][0]['qas'])

# print(len(data['data']))



