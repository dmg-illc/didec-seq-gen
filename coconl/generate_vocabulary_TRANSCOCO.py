import csv
import json
from collections import Counter
from collections import defaultdict
from utils.UtteranceTokenizer import UtteranceTokenizer
import string
import re

#VOCAB FROM TRAIN SPLIT of the translated COCO (in Dutch)
# generate coco vocab and combine with didec
# seems that all didec imgs are in train split
# also skip images in the didec val and test

# string vg ids
with open('../data/split_train.json', 'r') as file:
    didec_train_set = json.load(file)
with open('../data/split_val.json', 'r') as file:
    didec_val_set = json.load(file)
with open('../data/split_test.json', 'r') as file:
    didec_test_set = json.load(file)

# int coco ids
with open('../data/translated_coco/train_cap_img_ids_full.json', 'r') as file:
    tr_coco_img_ids = json.load(file)

with open('../data/translated_coco/captions_train2017_NL_full.json', 'r') as file:
    split_train = json.load(file)

with open('../data/translated_coco/val_cap_img_ids.json', 'r') as f:
    val_img_ids = json.load(f)

# converts Visual Genome ID to COCO ID
vg2coco = defaultdict(int)

with open("../data/imgs2.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        vg2coco[line[0]] = int(line[1])

train_keys = []
val_keys = []
test_keys = []

for k in didec_train_set:
    train_keys.append(vg2coco[k])

for k in didec_val_set:
    val_keys.append(vg2coco[k])

for k in didec_test_set:
    test_keys.append(vg2coco[k])

train_keys = set(train_keys)
val_keys = set(val_keys)
test_keys = set(test_keys)

all_text = []

tokenizer = UtteranceTokenizer('nl')
lowercase = True
method = 'nltk'
min_occurrence = 3

'''
in nltk 's included such as auto's not auto 's 
'''

count = 0

print(len(split_train))

dd_imgs = []

for s in range(len(split_train)):

    count+=1

    sent = split_train[s]
    train_img_id = tr_coco_img_ids[s]

    if count%10000 == 0:
        print(count)

    if train_img_id not in val_keys or train_img_id not in test_keys:
        # skip caption if img in val or test of didec

        tokenized_caption = tokenizer.tokenize_utterance(utterance=sent, method=method,
                                                           lowercase=lowercase)

        no_punc_caption = [x for x in tokenized_caption if not re.fullmatch('[' + string.punctuation + ' ' + ']+', x)]

        all_text.extend(no_punc_caption)

    if train_img_id in train_keys or train_img_id in val_keys or train_img_id in test_keys:
        dd_imgs.append(train_img_id) # 307 all images from train

dd_imgs = set(dd_imgs)
print(len(dd_imgs))
print(count)

dd_val = []
dd_real = []
for v in val_img_ids:
    if v in train_keys:
        dd_val.append(v)
    elif v in val_keys or v in test_keys:
        dd_real.append(v)

dd_val = set(dd_val)
dd_real = set(dd_real)
print('dd',len(dd_val), len(dd_real)) # 0, 0

token_counter = Counter(all_text).most_common()
print(token_counter)

tokens2write = defaultdict(int)

#add <unk> <pad> <start> <end>

tokens2write['<pad>'] = 0
tokens2write['<unk>'] = 1
tokens2write['<start>'] = 2
tokens2write['<end>'] = 3

vocab_count = 4

for t in token_counter:
    token, freq = t

    if freq < min_occurrence:
        pass #these are unks

    else:
        tokens2write[token] = vocab_count
        vocab_count += 1

print(vocab_count)

with open('../data/WORDMAP_cocoNL.json', 'w') as file:
    json.dump(tokens2write, file)


