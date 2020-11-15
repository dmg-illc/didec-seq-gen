import json
import pickle
import csv
import os
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as pim
from collections import defaultdict


ppn = 15
image_id = 498226
#ppn + image pair

#trial 96 for ppn50
#there exist lines where left eye and right have different events (blink vs. fx or sc)

#image_path = '../data/example_image_orig.jpg'
# image with original size to overlay the bottom-up features
# get from coco

#image_path = '../data/pizza.jpg'
image_path = '../data/example_image_orig.jpg'

sequential = True #if sequential = True, shows the scan path of gaze, otherwise a heatmap
fast = True #displays the scan path at a faster rate

normal_images_path ='../data/images'
bordered_images_path = '../data/images_bordered' #images resized and with a gray frame


with open('train36_imgid2idx.pkl', 'rb') as file:
    train_imgid2idx = pickle.load(file)

with open('val36_imgid2idx.pkl', 'rb') as file:
    val_imgid2idx = pickle.load(file)


train_bf = h5py.File('train36.hdf5', 'r')
val_bf = h5py.File('val36.hdf5', 'r')
print(train_bf.keys())

#<KeysViewHDF5 ['image_bb', 'image_features', 'spatial_features']>

image_bbs = train_bf['image_bb']
image_sf = train_bf['spatial_features']
image_f = train_bf['image_features']


# converts Visual Genome ID to COCO ID
vg2coco = defaultdict(int)

with open("../data/imgs2.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        vg2coco[int(line[0])] = int(line[1])


#get the index of image in the hdf5 file given COCO id (vg is stored in the didec files)
coco_id = vg2coco[image_id]
print(coco_id)

if coco_id in train_imgid2idx:
    print('t')

    h5_id = train_imgid2idx[coco_id]
    specific_img_bbs = train_bf['image_bb'][h5_id]

elif coco_id in val_imgid2idx:
    print('v')

    h5_id = val_imgid2idx[coco_id]
    specific_img_bbs = val_bf['image_bb'][h5_id]


def is_gaze_on_grey(x,y):
    '''
    GREY REGION
    x range
    0-205 1473-1679
    y range
    0-49 1000-1049
    '''

    greyFlag = False

    if x <= 205 or x >= 1473 or y <= 49 or y >= 1000:
        greyFlag = True

    return greyFlag

img = pim.imread(image_path)

plt.figure(0)

ax = plt.gca()
imgplot = plt.imshow(img)

for bbox in specific_img_bbs:

    #from https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/demo_vg.py
    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor='red', linewidth=1.5)
    )

plt.axis('off')
plt.savefig('bottomup.png')
plt.figure(1)

with open('../data/split_train.json', 'r') as file:
    train_set = json.load(file)

with open('../data/split_val.json', 'r') as file:
    val_set = json.load(file)

with open('../data/split_test.json', 'r') as file:
    test_set = json.load(file)

with open('../data/dict_gaze.json', 'r') as file:
    gaze_dict = json.load(file)

alignments_path = '../data/alignments'


# NO NEED FOR THESE FOR LOOPS IN PLOTTING ONE IMAGE

for root, subdirs, files in os.walk(bordered_images_path):

    for f in files:


        image_path = os.path.join(root, str(image_id) + '.bmp')

        img = pim.imread(image_path)

        ax = plt.gca()
        ax.xaxis.tick_top()
        imgplot = plt.imshow(img)

        eyescan = gaze_dict[str(ppn)][str(image_id)]


        first_ts = float(eyescan['timestamps'][0]) #subtract this from all the other ts to normalize

        red = False

        for e in range(len(eyescan['timestamps'])):

            current_timestamp = (float(eyescan['timestamps'][e])-first_ts)/1000000
            #normalized ts in seconds starting from 0

            # From the BeGaze manual:
            # The origin (0, 0) of the stimulus coordinate system is in the upper left corner of the stimulus

            eye_x = (float(eyescan['rxs'][e]) + float(eyescan['lxs'][e]))/2
            eye_y = (float(eyescan['rys'][e]) + float(eyescan['lys'][e]))/2


            if is_gaze_on_grey(eye_x, eye_y):
                print('Gaze on grey') #Gaze falling out of the image


            print(float(eyescan['rxs'][e]), float(eyescan['lxs'][e]), float(eyescan['rys'][e]), float(eyescan['lys'][e]))
            print(eye_x, eye_y)

            circle1 = plt.Circle((eye_x, eye_y), 5, color='b', alpha=0.5)
            ax.add_artist(circle1)


            if sequential:
                if fast and e%250 == 0: #sample rate 250
                    plt.pause(0.01)
                elif e%10 == 0:
                    plt.pause(0.01)
            else:
                #heatmap!
                pass



        plt.show()

        break