import json
from collections import defaultdict
import random
import os
from math import floor

train_percentage = 0.8
val_percentage = 0.1
test_percentage = 0.1

#IMAGE DESCRIPTION DATA
with open('../data/dict_caption_audio_wav.json', 'r') as file:
    dict_caption_audio = json.load(file) #caption path -> audio path

# WARNING ALIGNMENTS MAY NOT INCLUDE SOME OF THESE CAPTION-AUDIO PAIRS, 38 OF THEM ARE REMOVED

alignment_pairs = []

for root, dirs, files in os.walk('../data/alignments'):

    for f in files:

        # alignments that are not empty
        file_path = os.path.join(root, f)

        if os.stat(file_path).st_size != 0:

            f_split = f.split('.')[0].split('_')

            alignment_ppn = f_split[2]
            alignment_img = f_split[3]

            alignment_pairs.append((alignment_ppn, alignment_img))

#FREE VIEWING DATA
with open('../data/dict_free_view.json', 'r') as file:
    dict_free_view = json.load(file) #img_id to p_id

imgs = defaultdict(list)
p_desc = []


for caption_path in dict_caption_audio:
    split_cap = caption_path.split('/')[3].split('.')[0].split('_')

    p_id = split_cap[2]
    img_id = split_cap[3]

    #print(p_id, img_id)

    # don't add all, only add the ones with alignments:

    if (p_id, img_id) in alignment_pairs:
        imgs[img_id].append(p_id)
        p_desc.append(p_id)


print(len(imgs)) # 307

caption_count = 0

for i in imgs:
    count = len(imgs[i])

    caption_count += count

print(caption_count) #4548 (4586-38)


imgs_free = defaultdict()

p_free = []

for img_id in dict_free_view:
    imgs_free[img_id] = dict_free_view[img_id]

p_count = 0

for i in imgs_free:
    count = len(imgs_free[i])

    p_count += count

    p_free.extend(imgs_free[i])

print(p_count) # 4905


key_imgs_desc = imgs.keys()
key_imgs_free = imgs_free.keys()

print(len(key_imgs_free), len(key_imgs_desc)) # 307, 307

p_desc = set(p_desc)
p_free = set(p_free)

print(len(p_desc), len(p_free)) # 45, 48
print(len(p_desc-p_free)) # 45
print(len(p_free-p_desc)) # 48

#images are the same in both conditions, participants are mutually exclusive (3 fewer participants in desc)


#create splits

#shuffle randomly and take percentages into account
img_list = list(imgs.keys())
random.shuffle(img_list)

print(len(img_list))

len_list = len(img_list)

print(str(len_list*0.8), str(len_list*0.1))

train_ind = round(len_list*train_percentage) + 1
val_ind = train_ind + floor(len_list*val_percentage)
test_ind = val_ind + floor(len_list*test_percentage)

print(train_ind, val_ind, test_ind)

train_imgs = img_list[:train_ind]
val_imgs = img_list[train_ind:val_ind]
test_imgs = img_list[val_ind:]

print('image counts')
print(len(train_imgs), len(val_imgs), len(test_imgs))

train_set = defaultdict(list)
val_set = defaultdict(list)
test_set = defaultdict(list)


for i in train_imgs:
    train_set[i] = imgs[i]

for i in val_imgs:
    val_set[i] = imgs[i]

for i in test_imgs:
    test_set[i] = imgs[i]


print('set sizes in terms of images')
print(len(train_set), len(val_set), len(test_set))

train_size = 0
val_size = 0
test_size = 0

train_participants = []
val_participants = []
test_participants = []

train_cap_avg = []
val_cap_avg = []
test_cap_avg = []

for d in train_set:
    train_size += len(train_set[d])
    train_cap_avg.append((len(train_set[d])))
    train_participants.extend(train_set[d])

for d in val_set:
    val_size += len(val_set[d])
    val_cap_avg.append((len(val_set[d])))
    val_participants.extend(val_set[d])

for d in test_set:
    test_size += len(test_set[d])
    test_cap_avg.append((len(test_set[d])))
    test_participants.extend(test_set[d])

train_participants = set(train_participants)
val_participants = set(val_participants)
test_participants = set(test_participants)


print('set sizes in terms of captions')
print(train_size, val_size, test_size)

print('no of participants')
print(len(train_participants), len(val_participants), len(test_participants))

print('average no of caps per img')
print(sum(train_cap_avg)/len(train_cap_avg), sum(val_cap_avg)/len(val_cap_avg), sum(test_cap_avg)/len(test_cap_avg))

#CREATE THE JSON FILES

with open('../data/split_train.json', 'w') as file:
    json.dump(train_set, file)

with open('../data/split_val.json', 'w') as file:
    json.dump(val_set, file)

with open('../data/split_test.json', 'w') as file:
    json.dump(test_set, file)