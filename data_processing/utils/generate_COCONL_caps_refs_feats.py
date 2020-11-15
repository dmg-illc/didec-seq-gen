import json
from utils.UtteranceTokenizer import UtteranceTokenizer
import string
import re
import pickle
import random
from collections import defaultdict
import csv

# TRANSLATED COCO
# CREATING THE FINAL DATASET FILES FOR TRAINING THE MODELS
# CAPTIONS, CAPTION LENGTHS, REFERENCES, (MASKED) IMAGE FEATURES

tokenizer = UtteranceTokenizer('nl')
lowercase = True
method = 'nltk'
min_occurrence = 1

with open('../data/WORDMAP_union.json', 'r') as f:
    vocab_union = json.load(f)

with open('train36_imgid2idx.pkl', 'rb') as file:
    train_imgid2idx = pickle.load(file)

with open('val36_imgid2idx.pkl', 'rb') as file:
    val_imgid2idx = pickle.load(file)


# converts Visual Genome ID to COCO ID
vg2coco = defaultdict(int)

with open("../data/imgs2.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        vg2coco[line[0]] = int(line[1])

# string vg ids
with open('../data/split_train.json', 'r') as file:
    didec_train_set = json.load(file)
with open('../data/split_val.json', 'r') as file:
    didec_val_set = json.load(file)
with open('../data/split_test.json', 'r') as file:
    didec_test_set = json.load(file)


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


translated_val_coco_file = "../data/translated_coco/captions_val2017_NL.json"

with open(translated_val_coco_file, 'r') as f:
    translated_val_captions = json.load(f)

translated_train_coco_file = "../data/translated_coco/captions_train2017_NL_full.json"

with open(translated_train_coco_file, 'r') as f:
    translated_train_captions = json.load(f)


with open('../data/translated_coco/train_cap_img_ids_full.json', 'r') as f:
    train_img_ids = json.load(f)

skip_indices = []
skip_img_ids = []

# imgs in coco train but didec val or test

new_train_img_ids = []

for i in range(len(train_img_ids)):
    im = train_img_ids[i]
    if im in val_keys or im in test_keys:
        skip_indices.append(i)
        skip_img_ids.append(im)
    else:
        new_train_img_ids.append(im)

skip_img_ids = set(skip_img_ids)
print('skipping:', len(skip_img_ids), len(skip_indices))
train_img_ids = new_train_img_ids

#
with open('../data/translated_coco/val_cap_img_ids.json', 'r') as f:
    val_img_ids = json.load(f)

# TAKE HALF OF THE VAL AS TEST
# AND THE REST KEEP AS VAL

#25014
#12507

# train_img_ids has COCO ids listed
# train_imgid2idx is the index to that coco id in the hdf5 file

val_img_set = set(val_img_ids)

test_imgs = random.sample(val_img_set, int(len(val_img_set)/2))
val_imgs = val_img_set - set(test_imgs)
print(len(val_imgs), len(test_imgs))

translated_test_captions = []
test_img_ids = []

translated_val_captions_new = []
val_img_ids_new = []

for e in range(len(val_img_ids)):
    img_id_ = val_img_ids[e]

    if img_id_ in test_imgs:
        translated_test_captions.append(translated_val_captions[e])
        test_img_ids.append(val_img_ids[e])

    elif img_id_ in val_imgs:
        translated_val_captions_new.append(translated_val_captions[e])
        val_img_ids_new.append(val_img_ids[e])

translated_val_captions = translated_val_captions_new
val_img_ids = val_img_ids_new

with open('../data/train_ids_coconl.json', 'w') as f:
    json.dump(train_img_ids, f)
with open('../data/test_ids_coconl.json', 'w') as f:
    json.dump(test_img_ids, f)
with open('../data/val_ids_coconl.json', 'w') as f:
    json.dump(val_img_ids, f)


len_imgs = defaultdict()
for c in train_img_ids:
    if c in len_imgs:
        len_imgs[c] += 1
    else:
        len_imgs[c] = 1

set_len = []

for r in len_imgs:
    # print(len(r))
    set_len.append(len_imgs[r])

xx=[s for s in set_len if s >5]
print(xx)
set_len = set(set_len)
print(set_len)

len_imgs = defaultdict()
for c in val_img_ids:
    if c in len_imgs:
        len_imgs[c] += 1
    else:
        len_imgs[c] = 1

set_len = []

for r in len_imgs:
    # print(len(r))
    set_len.append(len_imgs[r])

xx=[s for s in set_len if s >5]
print(xx)
set_len = set(set_len)
print(set_len)

len_imgs = defaultdict()
for c in test_img_ids:
    if c in len_imgs:
        len_imgs[c] += 1
    else:
        len_imgs[c] = 1

set_len = []

for r in len_imgs:
    # print(len(r))
    set_len.append(len_imgs[r])

xx=[s for s in set_len if s >5]
print(xx)
set_len = set(set_len)
print(set_len)
print()


###
# CAPTIONS - CAPTION LENGTHS

def create_coco_caps_data(coco_captions, vocab, skip, skip_indices, skip_imgs):
    captions = []
    caplens = []

    max_len = 0

    for c in range(len(coco_captions)):

        # if we are skipping things and the index is not in the skipping list
        # or we are simply not skipping things

        if (skip and c not in skip_indices) or not skip:
            utt = coco_captions[c]

            tokenized_caption = tokenizer.tokenize_utterance(utterance=utt, method=method,
                                                             lowercase=lowercase)

            no_punc_caption = [x for x in tokenized_caption if not re.fullmatch('[' + string.punctuation + ' ' + ']+', x)]

            caption2id = [vocab['<start>']]

            for token in no_punc_caption:

                if token in vocab:
                    token2id = vocab[token]
                else:
                    token2id = vocab['<unk>']

                caption2id.append(token2id)

            caption2id.append(vocab['<end>'])

            if len(caption2id) > max_len:
                max_len = len(caption2id)

            captions.append(caption2id)
            caplens.append(len(caption2id))

    print(skip, len(captions))
    print(max_len)
    for c in captions:
        if len(c) < max_len:
            pad_dif = max_len - len(c)

            for p in range(pad_dif):
                c.append(vocab['<pad>'])

    return captions, caplens

skip_flag = True
captions_train, caplens_train = create_coco_caps_data(translated_train_captions, vocab_union, skip_flag, skip_indices, skip_img_ids)

skip_flag = False # nothing to skip here
captions_val, caplens_val = create_coco_caps_data(translated_val_captions, vocab_union, skip_flag, skip_indices, skip_img_ids)
captions_test, caplens_test = create_coco_caps_data(translated_test_captions, vocab_union, skip_flag, skip_indices, skip_img_ids)

#maxlen tr,val,ts
#54
#38
#45 now

# captions, list of lists
with open('../data/train_CAPTIONS_coconl.json', 'w') as file:
    json.dump(captions_train, file)

with open('../data/val_CAPTIONS_coconl.json', 'w') as file:
    json.dump(captions_val, file)

with open('../data/test_CAPTIONS_coconl.json', 'w') as file:
    json.dump(captions_test, file)

# caption lengths, lists
with open('../data/train_CAPLENS_coconl.json', 'w') as file:
    json.dump(caplens_train, file)

with open('../data/val_CAPLENS_coconl.json', 'w') as file:
    json.dump(caplens_val, file)

with open('../data/test_CAPLENS_coconl.json', 'w') as file:
    json.dump(caplens_test, file)

#####
#####
# given a coco img id RETURNS which split and where in the bottom-up features it is located [tr or val, idx]
# we need to do this because images in our case might belong to different splits

# dict might be better, now a list
train_FEATS_coconl = []
val_FEATS_coconl = []
test_FEATS_coconl = []

for s in train_img_ids:

    # already skipped didec val and test imgs
    if s in train_imgid2idx:
        train_FEATS_coconl.append(['t', train_imgid2idx[s]])
    elif s in val_imgid2idx:
        train_FEATS_coconl.append(['v', val_imgid2idx[s]])

    else:
        print('tr', s)

for s in val_img_ids:

    if s in train_imgid2idx:
        val_FEATS_coconl.append(['t', train_imgid2idx[s]])
    elif s in val_imgid2idx:
        val_FEATS_coconl.append(['v', val_imgid2idx[s]])

    else:
        print('val', s)

for s in test_img_ids:

    if s in train_imgid2idx:
        test_FEATS_coconl.append(['t', train_imgid2idx[s]])
    elif s in val_imgid2idx:
        test_FEATS_coconl.append(['v', val_imgid2idx[s]])

    else:
        print('ts', s)


with open('../data/train_FEATS_coconl.json', 'w') as file:
    json.dump(train_FEATS_coconl, file)

with open('../data/val_FEATS_coconl.json', 'w') as file:
    json.dump(val_FEATS_coconl, file)

with open('../data/test_FEATS_coconl.json', 'w') as file:
    json.dump(test_FEATS_coconl, file)


####
# CAPTION REFERENCES

allrefs_train = dict()
allrefs_val = dict()
allrefs_test = dict()

for f in range(len(train_img_ids)):
    # already skipped didec val and test

    img_idx = train_img_ids[f]

    if img_idx in allrefs_train:
        allrefs_train[img_idx].append(f)
    else:
        allrefs_train[img_idx] = [f]

for f in range(len(val_img_ids)):
    img_idx = val_img_ids[f]

    if img_idx in allrefs_val:
        allrefs_val[img_idx].append(f)
    else:
        allrefs_val[img_idx] = [f]

for f in range(len(test_img_ids)):
    img_idx = test_img_ids[f]

    if img_idx in allrefs_test:
        allrefs_test[img_idx].append(f)
    else:
        allrefs_test[img_idx] = [f]

def get_refs(captions, allrefs):

    refs4cap = []

    for c in range(len(captions)):

        if c % 10000 == 0:
            print(c)

        for ix in allrefs:

            refs = allrefs[ix]

            if c in refs:
                #print(ix, c, refs)

                ref_caps  = []

                for r in refs:
                    ref_caps.append(captions[r])

                refs4cap.append(ref_caps)

                break

    return refs4cap

print('train')
train_refs = get_refs(captions_train, allrefs_train)

print('val')
val_refs = get_refs(captions_val, allrefs_val)

print('test')
test_refs = get_refs(captions_test, allrefs_test)


with open('../data/train_REFS_coconl.json', 'w') as file:
    json.dump(train_refs, file)

with open('../data/val_REFS_coconl.json', 'w') as file:
    json.dump(val_refs, file)

with open('../data/test_REFS_coconl.json', 'w') as file:
    json.dump(test_refs, file)
