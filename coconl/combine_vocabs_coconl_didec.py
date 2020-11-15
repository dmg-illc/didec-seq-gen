import json
from collections import defaultdict
import string

# create a combined vocab of didec and translated COCO

union_vocab = defaultdict(int)

with open('../data/WORDMAP_didec.json', 'r') as f:
    didec_vocab = json.load(f)

with open('../data/WORDMAP_cocoNL.json', 'r') as f:
    coconl_vocab = json.load(f)

difs = set(coconl_vocab.keys()) - set(didec_vocab.keys())
difs2 = set(didec_vocab.keys()) - set(coconl_vocab.keys())
print(len(difs))
print(len(difs2))
print(difs2)

#58838
#780
# union 62632 coconl 61852 didec 3794

union_vocab = didec_vocab.copy()

ind_counter = len(union_vocab)

for k in coconl_vocab.keys():
    if k not in union_vocab.keys():
        union_vocab.update({k:ind_counter})
        ind_counter += 1

print('union', len(union_vocab), 'coconl', len(coconl_vocab), 'didec', len(didec_vocab))


with open('../data/WORDMAP_union.json', 'w') as f:
    json.dump(union_vocab, f)
