import json
from collections import defaultdict
import csv

#COCO instances to get the object annotations (numbers, categories, areas, locations)

with open('../data/coco_inst/instances_val2017.json', 'r') as file:
    val_coco_instances = json.load(file)

val_annots = val_coco_instances['annotations']

with open('../data/coco_inst/instances_train2017.json', 'r') as file:
    train_coco_instances = json.load(file)

train_annots = train_coco_instances['annotations']

# converts Visual Genome ID to COCO ID
vg2coco = defaultdict(int)

with open("../data/imgs2.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        vg2coco[int(line[0])] = int(line[1])

with open('../data/split_train.json', 'r') as file:
    train_set = json.load(file)

with open('../data/split_val.json', 'r') as file:
    val_set = json.load(file)

with open('../data/split_test.json', 'r') as file:
    test_set = json.load(file)


all_imgs = []

all_imgs.extend(list(train_set.keys()))
all_imgs.extend(list(test_set.keys()))
all_imgs.extend(list(val_set.keys()))

#for DIDEC images:
img2objs = defaultdict(list)

for img_id in all_imgs:

    coco_id = vg2coco[int(img_id)]

    for a in train_annots:
        if a['image_id'] == coco_id:
            img2objs[coco_id].append(a)

    for a in val_annots:
        if a['image_id'] == coco_id:
            img2objs[coco_id].append(a)

print(len(img2objs))

with open('../data/didec_img2objs.json', 'w') as file:
    json.dump(img2objs, file)


