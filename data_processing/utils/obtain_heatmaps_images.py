import json
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as pim
from scipy import ndimage
import numpy as np

# see a visual example in plot_heatmap_example.py
'''
https://tomekloboda.net/res/research/deg-to-pix/
https://osdoc.cogsci.nl/3.2/visualangle/
https://res18h39.netlify.com/calculator

Distance from the screen (D)
700 mm 
Screen diameter(d)
22 inches
Horizontal screen resolution(sx)
1680 pixels
Vertical screen resolution(sy)
1050 pixels
Fixation radius(α)
1 degrees

Fixation radius(p)
43.32 pixels

***
45.52 from convert_degree2pix.py
'''

sigma = 44 #to increase the gaussian area around fixations

#https://www.mindmetriks.com/uploads/4/4/6/0/44607631/smi_flyer_red250mobile.pdf
#GAZE POZITION ACCURACY 0.4 DEGREES SMI RED 250

duration_weighted = True # weights fixation areas by normalized duration per participant-image pair

draw_details = False

task = 'free_view'
#task = 'description

# no of total fixations in DS 142279
# no of total fixations in FV 48990

if task == 'description':
    with open('../data/fixation_events_DS.json', 'r') as file:
        fixation_events = json.load(file)

elif task == 'free_view':
    with open('../data/fixation_events_FV.json', 'r') as file:
        fixation_events = json.load(file)

masked_images_path = '../data/masked_images/'

if not os.path.isdir(masked_images_path):
    os.mkdir(masked_images_path)

masked_images_path += task

if not os.path.isdir(masked_images_path):
    os.mkdir(masked_images_path)


#https://chsasank.github.io/vision/models
#All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
#where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using
#mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_fixation_window(fixation, ax, plt, draw_details):

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
    duration = (max_t - min_t)  #all fixation based, not related to the actual recording duration
    # beginning of first fixation and end of last fixation

    #print(min_x, min_y, width, height)

    centroid = ((min_x + width/2), (min_y + height/2)) # eq to (minx + maxx) / 2

    if draw_details:
        ax.add_patch(plt.Rectangle((min_x, min_y), width, height, fill=True, alpha=0.5,
                                   color='cyan', linewidth=1.5))

        circle = plt.Circle(centroid, 1, color='r', alpha=1)
        ax.add_artist(circle)

    return centroid, duration

def create_heatmap_data_per_image(sigma, duration_weighted):

    #for each image creates an aggregated heatmap over all the participants who looked at that image

    count = 0

    img2mask = dict()  # per image, aggregates all participants' heatmaps

    # ppn 11 fix maps for img 713519 empty NO FIXATIONS on the actual image

    ppn_count = 0

    for ppn in fixation_events:

        ppn_count += 1
        print(ppn_count)

        for img_id in fixation_events[ppn]:

            count += 1

            print(count, img_id)

            fixation_windows = fixation_events[ppn][img_id]

            grey_img_file = '../data/images_bordered/' + img_id + '.bmp'

            img = pim.imread(grey_img_file)

            image = Image.open(grey_img_file).convert('RGB')

            plt.figure(0)

            plt.axis('off')

            ax = plt.gca()

            width = image.size[0]
            height = image.size[1]

            final_heatmap = np.zeros((height, width)) #1050, 1680 with borders reverse in np

            durations = []
            fixation_heatmaps = []

            for f in fixation_windows:

                centroid, duration = get_fixation_window(f, ax, plt, draw_details)

                #generate a gaussian around the centroid over the whole image

                x = np.zeros((height, width))
                #print(x.shape)

                # not the precise centroid, rather an approximation due to the discreteness of image pixels
                centroid = (round(centroid[1]), round(centroid[0])) #reverse in np

                x[centroid] = 1

                # https://github.com/durandtibo/heatmap/blob/master/heatmap/heat_map.py
                fixation_heatmap = ndimage.filters.gaussian_filter(x, sigma = sigma)
                #print(fixation_heatmap.shape)

                durations.append(duration)

                fixation_heatmaps.append(fixation_heatmap)

                #print(centroid)

            # normalize the durations in the context of this image:
            # RETHINK THE DURATIONS WHEN WE SKIP THINGS

            durations = [d/sum(durations) for d in durations]

            #print(durations)

            for fh in range(len(fixation_heatmaps)):

                if duration_weighted:
                    final_heatmap += durations[fh] * fixation_heatmaps[fh]
                else:
                    final_heatmap += fixation_heatmaps[fh]

            max_value = np.max(final_heatmap)
            min_value = np.min(final_heatmap)

            if max_value != 0:
                normalized_heatmap = (final_heatmap - min_value) / (max_value - min_value)

                if img_id in img2mask:
                    img2mask[img_id] += normalized_heatmap
                else:
                    img2mask[img_id] = [normalized_heatmap]

    img_count = 0

    for img_id in img2mask:

        img_count += 1

        print('img', img_count)

        combined_heatmap = img2mask[img_id] #((height, width)) #1050, 1680 with borders reverse in np

        max_v = np.max(combined_heatmap)
        min_v = np.min(combined_heatmap)
        normalized_combined_heatmap = (combined_heatmap - min_v) / (max_v - min_v)

        grey_img_file = '../data/images_bordered/' + img_id + '.bmp'

        img = pim.imread(grey_img_file)

        #image = Image.open(grey_img_file).convert('RGB')

        new_img = img.copy()

        # mask the image given the heatmap
        new_img[:,:,0] = new_img[:,:,0] * normalized_combined_heatmap
        new_img[:,:,1] = new_img[:,:,1] * normalized_combined_heatmap
        new_img[:,:,2] = new_img[:,:,2] * normalized_combined_heatmap

        #plt.imshow(new_img)

        # CROP the masked image and save it so that we can obtain its features
        masked_image = Image.fromarray(new_img, 'RGB')
        masked_image = masked_image.crop((206, 50, 206 + 1267, 50 + 950)) # crop to the size of the actual image on the screen
        masked_img_path = os.path.join(masked_images_path, 'masked_' + img_id + '.png')
        masked_image.save(masked_img_path)

        # also save the mask itself
        mask = Image.fromarray((normalized_heatmap, 'RGB'))
        mask = mask.crop((206, 50, 206 + 1267, 50 + 950))
        mask_path = os.path.join((masked_img_path,  'masks', 'mask_' + img_id + '.png'))
        mask.save(mask_path)

    print('count', count)

create_heatmap_data_per_image(sigma, duration_weighted)