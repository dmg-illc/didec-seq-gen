import json
from collections import defaultdict
import csv

check_type = "bbox" # centroid
file_name = '../data/scanpaths_DS_' + check_type + '.json'

with open(file_name, 'r') as f:
    scanpaths_dict = json.load(f)

with open('../data/split_train.json', 'r') as file:
    train_set = json.load(file)

with open('../data/split_val.json', 'r') as file:
    val_set = json.load(file)

with open('../data/split_test.json', 'r') as file:
    test_set = json.load(file)

# converts Visual Genome ID to COCO ID
vg2coco = defaultdict(int)

with open("../data/imgs2.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        vg2coco[line[0]] = int(line[1])

no_of_obj_per_fx = 0
no_of_fx = 0

no_of_obj_per_fx_no_empty = 0
no_of_fx_no_empty = 0

no_of_obj_per_cap = 0
no_of_cap = 0 # total with repetition

no_of_empty_fx = 0

completely_empty_scanpath = 0

for p in scanpaths_dict:

    for img in scanpaths_dict[p]:

        no_of_cap += 1
        sp = scanpaths_dict[p][img]

        completely_empty_flag = True

        for f in range(len(sp)):
            fx_objs = sp[f]

            no_of_fx += 1

            if len(fx_objs) > 0:

                completely_empty_flag = False

                no_of_obj_per_fx += len(fx_objs)
                no_of_obj_per_cap += len(fx_objs)

                no_of_obj_per_fx_no_empty += len(fx_objs)
                no_of_fx_no_empty += 1

            else:
                no_of_empty_fx += 1


        if completely_empty_flag:
            completely_empty_scanpath += 1


no_of_obj_per_fx = no_of_obj_per_fx / no_of_fx
no_of_obj_per_cap = no_of_obj_per_cap / no_of_cap
no_of_obj_per_fx_no_empty = no_of_obj_per_fx_no_empty / no_of_fx_no_empty

print(check_type)
print(no_of_obj_per_fx, no_of_fx)
print(no_of_obj_per_fx_no_empty, no_of_fx_no_empty)
print(no_of_obj_per_cap, no_of_cap)
print(no_of_empty_fx)
print(completely_empty_scanpath)




