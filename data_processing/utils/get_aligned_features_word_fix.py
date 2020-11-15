import json
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms
from PIL import Image
import matplotlib.image as pim
from scipy import ndimage
import numpy as np

# break after 1 generation, as an example
# some code overlaps with obtain_heatmaps_images.py

with open('../data/aligned_fixations_words.json', 'r') as file:
    alignments = json.load(file)

sigma = 44
duration_weighted = True

# bottom up features come from this
resnet101 = models.resnet101(pretrained=True)
print(resnet101)
modules = list(resnet101.children())[:-1]
resnet101 = nn.Sequential(*modules)
resnet101.eval()  # no fine-tuning


#transformations required by resnet
img_transform = transforms.Compose([
    transforms.Resize((224,224)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#https://pytorch.org/docs/stable/torchvision/models.html
#All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
#where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using
#mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].


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

    centroid = ((min_x + width/2), (min_y + height/2)) # eq to (minx + maxx) / 2

    return centroid, duration

def get_masked_image_features(img_id, fixations, model, duration_weighted, img_transform, count):

    grey_img_file = '../data/images_bordered/' + img_id + '.bmp'

    img = pim.imread(grey_img_file)

    image = Image.open(grey_img_file).convert('RGB')

    width = image.size[0]
    height = image.size[1]

    final_heatmap = np.zeros((height, width))  # 1050, 1680 with borders reverse in np

    durations = [] # in this model, only normalized among fixations for one word
    fixation_heatmaps = []

    count += 1

    for f in fixations:
        centroid, duration = get_fixation_window(f)

        # generate a gaussian around the centroid over the whole image

        x = np.zeros((height, width))
        # print(x.shape)

        # not the precise centroid, rather an approximation due to the discreteness of image pixels
        centroid = (round(centroid[1]), round(centroid[0]))  # reverse in np

        x[centroid] = 1

        # https://github.com/durandtibo/heatmap/blob/master/heatmap/heat_map.py
        fixation_heatmap = ndimage.filters.gaussian_filter(x, sigma=sigma)
        # print(fixation_heatmap.shape)

        durations.append(duration)

        fixation_heatmaps.append(fixation_heatmap)

    if len(durations) > 0:
        # at least one window, but that window can have one or multiple gazes
        # in the case of one gaze (most likely some noise, but I still use them
        # duration is 0, so check for that as well
        # only one gaze in the whole fixation window (because I removed blinks and out-of-image gazes
        # set that duration to a very small number (the smallest positive which is not eq to 0

        durations[durations==0] = 1e-323 # so the sum is never exactly 0

        durations = [d / sum(durations) for d in durations]
        # normalized among a set of fixation for one word only

        #print(durations)

        for fh in range(len(fixation_heatmaps)):

            if duration_weighted:
                final_heatmap += durations[fh] * fixation_heatmaps[fh]
            else:
                final_heatmap += fixation_heatmaps[fh]

        max_value = np.max(final_heatmap)
        min_value = np.min(final_heatmap)

        if max_value != 0:
            final_heatmap = (final_heatmap - min_value) / (max_value - min_value)


    # if len(durations) == 0, no fixations, final_heatmap is all zeros
    # we end up with a black picture for the related word

    new_img = img.copy()

    # mask the image given the heatmap
    new_img[:, :, 0] = new_img[:, :, 0] * final_heatmap
    new_img[:, :, 1] = new_img[:, :, 1] * final_heatmap
    new_img[:, :, 2] = new_img[:, :, 2] * final_heatmap

    # plt.imshow(new_img)

    # CROP the masked image and so that we can obtain its features
    image = Image.fromarray(new_img, 'RGB')
    image = image.crop((206, 50, 206 + 1267, 50 + 950))  # crop to the size of the actual image on the screen

    # apply necessary transformations for resnet
    print('COUNT', count)

    image.save('masked_align_' + img_id + '_' + ppn  + '_' + str(count) + '.png')

    image = img_transform(image)

    input = Variable(image).view(1, image.shape[0], image.shape[1], image.shape[2])
    output = model(input)
    output = output[0].squeeze(1).squeeze(1).data

    masked_image_features = output.numpy().tolist()

    return masked_image_features, count


count = 0

for ppn in alignments:

    for img_id in alignments[ppn]:

        if img_id == '1591977': # example for the thesis
            print(ppn, img_id)

            single_alignment = alignments[ppn][img_id]

            single_fxs = []

            count = 0

            for alg in single_alignment:

                token = alg[0]
                fixations = alg[1]
                print(token)

                #count += len(fixations) # TOTAL COUNT OF FX IS DIFF FROM DS, as I don't have the final gazes at the end

                #print(token, fixations)

                masked_img_features, count = get_masked_image_features(img_id, fixations, resnet101, duration_weighted, img_transform, count)

                single_fxs.append((token, masked_img_features))

            break



        #with open('../data/aligned_features_fixations_words_' + ppn + '_' + img_id + '.json', 'w') as file:
            #json.dump(single_fxs, file)
