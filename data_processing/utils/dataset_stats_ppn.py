import json
from collections import defaultdict

with open('../data/split_train.json', 'r') as f:
    split_train = json.load(f)

with open('../data/split_test.json', 'r') as f:
    split_test = json.load(f)

with open('../data/split_val.json', 'r') as f:
    split_val = json.load(f)


ppn2imgs = defaultdict(list)

for img in split_train:

    ppns4img = split_train[img]

    for ppn in ppns4img:

        ppn2imgs[ppn].append(img)

for img in split_test:

    ppns4img = split_test[img]

    for ppn in ppns4img:
        ppn2imgs[ppn].append(img)

for img in split_val:

    ppns4img = split_val[img]

    for ppn in ppns4img:
        ppn2imgs[ppn].append(img)


print(len(ppn2imgs.keys()))

no_imgs4ppn = []

for ppn in ppn2imgs:

    no_imgs4ppn.append(len(ppn2imgs[ppn]))

print(no_imgs4ppn)

print(len(no_imgs4ppn))

avg_imgs4ppns = sum(no_imgs4ppn) / len(no_imgs4ppn)

print(avg_imgs4ppns)