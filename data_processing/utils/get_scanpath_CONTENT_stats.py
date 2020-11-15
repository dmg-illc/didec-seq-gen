import json
from collections import defaultdict
import csv

check_type = "bbox" # centroid
file_name = '../data/scanpaths_DS_CONTENT_' + check_type + '.json'

with open('../data/CONTENT_captions.json', 'r') as file:
    content_captions = json.load(file)

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
no_of_fx = 0 # NO OF FX GROUPS

no_of_obj_per_fx_no_empty = 0
no_of_fx_no_empty = 0

no_of_obj_per_cap = 0
no_of_cap = 0 # total with repetition

no_of_empty_fx = 0

completely_empty_scanpath = 0

for p in scanpaths_dict:

    for img in scanpaths_dict[p]:

        print(content_captions[p][img])

        content_caption_text = content_captions[p][img]
        no_of_cap += 1
        sp = scanpaths_dict[p][img]

        for c_i in range(len(content_caption_text)):
            print(content_caption_text[c_i])
            print(sp[c_i])

        completely_empty_flag = True

        for f in range(len(sp)):
            fx_objs = sp[f]

            if len(fx_objs) > 0:

                completely_empty_flag = False

                for s_fx_obj in fx_objs:
                    no_of_fx += 1

                    if len(s_fx_obj) > 0:
                        no_of_obj_per_fx += len(s_fx_obj)
                        no_of_obj_per_cap += len(s_fx_obj)
                        no_of_obj_per_fx_no_empty += len(s_fx_obj)
                        no_of_fx_no_empty += 1

                    else:
                        no_of_empty_fx += 1


        if completely_empty_flag:
            completely_empty_scanpath += 1


print(no_of_obj_per_fx, no_of_obj_per_cap, no_of_obj_per_fx_no_empty)

no_of_obj_per_fx = no_of_obj_per_fx / no_of_fx
no_of_obj_per_cap = no_of_obj_per_cap / no_of_cap
no_of_obj_per_fx_no_empty = no_of_obj_per_fx_no_empty / no_of_fx_no_empty

print(no_of_obj_per_fx, no_of_obj_per_cap, no_of_obj_per_fx_no_empty)

print(check_type)
print(no_of_obj_per_fx, no_of_fx)
print(no_of_obj_per_fx_no_empty, no_of_fx_no_empty)
print(no_of_obj_per_cap, no_of_cap)
print(no_of_empty_fx)
print(completely_empty_scanpath)



'''
bbox
1.2619057072748496 125927
1.6780502228135759 94698
34.94019349164468 4548
31229
1

centroid
1.0452722609130687 125927
1.4896448699667277 88362
28.941952506596305 4548
37565
1

'''


