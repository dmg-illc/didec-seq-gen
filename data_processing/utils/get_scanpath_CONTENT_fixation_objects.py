# this will operate on the processed fx events of description view (not the raw from DIDEC, not the window ones aligned with words)
#

import json
from collections import defaultdict
import csv

# also projecting coco coordinates to didec image dims

check_type = "centroid" # 'centroid' or 'bbox' to check intersection with coco objects

# PROJECT TO  h 950 w 1267 from h 480 w 640 for coco bbox
# new/old
ratio_x = 1267/640 # 1.9796875
ratio_y = 950/480 # 1.9791666666666667

def get_fixation_window(fixation):

    # THIS IS DIFFERENT FROM THE PREVIOUS CODE THAT LOOKS AT EVERY WORD (not just content words)

    # FOR A SINGLE CONTENT WORD

    centroids = []
    durations = []
    fixation_bbs = []

    # fx_group
    # in fx_group there are multiple fixation items FOR A SINGLE CONTENT WORD

    for fx_group in fixation:

        all_xs = []
        all_ys = []
        all_ts = []

        for gaze_item in fx_group:

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

        # a la coco bbox

        # coco bbox
        # [x,y,width,height]
        # an enclosing bounding box is provided for each object (box coordinates are measured from the top left image corner and are 0-indexed).

        # INCLUDES GREY BORDERS!!
        fixation_bb = (min_x, min_y, width, height)

        centroids.append(centroid)
        durations.append(duration)
        fixation_bbs.append(fixation_bb)

    return centroids, durations, fixation_bbs

def project_bbox(bbox, ratio_x, ratio_y):

    # PROJECT TO  h 950 w 1267 from h 480 w 640 for coco bbox
    top_x, top_y, width, height = bbox

    top_x = top_x * ratio_x
    top_y = top_y * ratio_y

    width = width * ratio_x
    height = height * ratio_y

    projected_bbox = (top_x, top_y, width, height)
    return projected_bbox

def in_bbox(bbox_coco, centroid, fix_bbox, check_type):
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

    # PROJECT TO  h 950 w 1267 from h 480 w 640 for coco bbox

    prj_bbox = project_bbox(bbox_coco, ratio_x, ratio_y)

    top_x, top_y, width, height = prj_bbox

    bbox_flag = False

    if check_type == 'centroid':
        c_x, c_y = centroid

        c_x -= 206
        c_y -= 50

        if c_x >= top_x and c_x <= (top_x + width) and c_y >= top_y and c_y <= (top_y + height):
            bbox_flag = True

    elif check_type == 'bbox':
        f_top_x, f_top_y, f_width, f_height = fix_bbox

        f_top_x -= 206
        f_top_y -= 50
        # from https://stackoverflow.com/a/13390495
        # if (X1 + W1 < X2 or X2 + W2 < X1 or Y1 + H1 < Y2 or Y2 + H2 < Y1):
        #     Intersection = Empty
        # else:
        #     Intersection = Not Empty

        if (top_x + width) < f_top_x or (f_top_x + f_width) < top_x or (top_y + height) < f_top_y or (f_top_y + f_height) < top_y:
            bbox_flag = False
        else:
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

# NOT THE FULL SET OF FIXATIONS
# with open('../data/fixation_events_DS.json', 'r') as file:
#     fixations_dict = json.load(file)

# FIXATION GROUPS ALIGNED WRT CONTENT WORDS ONLY
with open('../data/aligned_fixations_CONTENT_words.json', 'r') as file:
    fixations_dict = json.load(file)

print(len(fixations_dict))

with open('../data/CONTENT_captions.json', 'r') as file:
    content_captions = json.load(file)

with open('../data/annotations_trainval2014/annotations/instances_train2014.json', 'r') as file:
    train_coco_instances = json.load(file)


coco_id2cat = defaultdict()

for cts in train_coco_instances['categories']:
    print(cts)

    coco_id2cat[cts['id']] = [cts['name']]


# now using the fixation windows compare details

count_p = 0

scanpaths_dict = defaultdict()

for p in fixations_dict:

    count_p += 1

    scanpaths_dict_ppn = defaultdict()

    for img in fixations_dict[p]:

        print('p', p, 'im', img, count_p)
        print(content_captions[p][img])

        coco_id = str(vg2coco[int(img)])

        objs = img2objs[coco_id]

        fixation_windows = fixations_dict[p][img]

        scanpath = []

        for w in fixation_windows:

            content_word = w[0] # FOR A SINGLE CONTENT WORD
            content_fxs = w[1]

            # THIS IS DIFFERENT FROM PREV CODE, RETURNS MULTIPLE VALUES PER CONTENT WORDS
            centres, durs, fix_bbs = get_fixation_window(content_fxs)

            # if len(content_fxs) != len(centres) or len(w) != len(fix_bbs):
            #     print(len(content_fxs), len(centres), len(fix_bbs))

            fx_scanpath = []

            for fg in range(len(centres)):

                objs_per_group = []

                #for each group for a single content word, check separately
                centre = centres[fg]
                fix_bb = fix_bbs[fg]

                for o in objs:

                    is_in_bbox = in_bbox(o['bbox'], centre, fix_bb, check_type)

                    if is_in_bbox:
                        #print('OBJ', o['category_id'], o['id'])

                        objs_per_group.append((o['id'], coco_id2cat[o['category_id']][0])) #unique obj id and category id of the obj

                fx_scanpath.append(objs_per_group)

            scanpath.append(fx_scanpath) # per word

        print(scanpath)

        scanpaths_dict_ppn[img] = scanpath

    scanpaths_dict[p] = scanpaths_dict_ppn

file_name = '../data/scanpaths_DS_CONTENT_' + check_type + '.json'
with open(file_name, 'w') as f:
    json.dump(scanpaths_dict, f)