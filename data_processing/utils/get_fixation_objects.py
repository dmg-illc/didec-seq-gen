# this will operate on the processed fx events of description view (not the raw from DIDEC, not the window ones aligned with words)
#

import json
from collections import defaultdict
import csv

def get_fixation_window(fixation):

    all_xs = []
    all_ys = []
    all_ts = []

    for gaze_item in fixation:

        # LEFT AND RIGHT POSITIONS ARE THE SAME
        #gaze_item = (timestamp,l_por_x, l_por_y, r_por_x, r_por_y)
        # 1,2 or 3,4

        all_xs.append(float(gaze_item[3]))
        all_ys.append(float(gaze_item[4]))
        all_ts.append(float(gaze_item[0]))

    min_x = min(all_xs)
    max_x = max(all_xs)
    min_y = min(all_ys)
    max_y = max(all_ys)

    width = max_x - min_x
    height = max_y - min_y

    min_t = min(all_ts)
    max_t = max(all_ts)
    duration = (max_t - min_t)

    if max_y > 1049:

        print('maxxx')
        print()

    centroid = ((min_x + width/2), (min_y + height/2)) # eq to (minx + maxx) / 2

    return centroid, duration


def in_bbox(bbox_coco, centroid):
    #http: // cocodataset.org /  # format-data
    # also didec has grey borders
    # adjust the coordinates
    #image = image.crop((206, 50, 206 + 1267, 50 + 950))  # crop to the size of the actual image on the screen

    '''
    GREY REGION
    x range
    0-205 1473-1679
    y range
    0-49 1000-1049
    '''

    top_x, top_y, width, height = bbox_coco
    c_x, c_y = centroid

    c_x -= 206
    c_y -= 50

    bbox_flag = False

    if c_x >= top_x and c_x <= (top_x + width) and c_y >= top_y and c_y <= (top_y + height):
        bbox_flag = True

    return bbox_flag


#compare fixation windows given COCO object annotations (numbers, categories, areas, locations)

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

# coco bbox
# [x,y,width,height]
# an enclosing bounding box is provided for each object (box coordinates are measured from the top left image corner and are 0-indexed).


#contains all images from 3 splits
with open('../data/didec_img2objs.json', 'r') as file:
    img2objs = json.load(file)

#use aligned fixation gaze event dicts
#left eye event might differ from right eye event
#even then, the x-y positions for L and R are the same
#includes all splits

with open('../data/fixation_events_DS.json', 'r') as file:
    fixations_dict = json.load(file)

print(len(fixations_dict))

# now using the fixation windows compare details

count_p = 0

scanpaths_dict = defaultdict()

for p in fixations_dict:

    count_p += 1

    scanpaths_dict_ppn = defaultdict()

    for img in fixations_dict[p]:

        print('p', p, 'im', img, count_p)

        coco_id = str(vg2coco[int(img)])

        objs = img2objs[coco_id]

        fixation_windows = fixations_dict[p][img]

        scanpath = []

        for w in fixation_windows:

            centre, dur = get_fixation_window(w)

            #print(centre)

            for o in objs:

                is_in_bbox = in_bbox(o['bbox'], centre)

                if is_in_bbox:
                    #print('OBJ', o['category_id'], o['id'])

                    scanpath.append((o['id'], o['category_id'])) #unique obj id and category id of the obj


        #print(scanpath)

        scanpaths_dict_ppn[img] = scanpath

    scanpaths_dict[p] = scanpaths_dict_ppn


with open('../data/scanpaths_DS.json', 'w') as f:
    json.dump(scanpaths_dict, f)