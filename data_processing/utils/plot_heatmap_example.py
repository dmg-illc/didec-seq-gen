import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pim
from PIL import Image
import json
from scipy import ndimage

sigma = 44 # 1 degree
# #to increase the gaussian area around fixations

#https://www.mindmetriks.com/uploads/4/4/6/0/44607631/smi_flyer_red250mobile.pdf
#GAZE POZITION ACCURACY 0.4 DEGREES SMI RED 250

duration_weighted = True # weights fixation areas by normalized duration per participant-image pair

ppn_id = '15'
img_id = '498226'

draw_details = True

task = 'description'

if task == 'description':
    with open('../data/fixation_events_DS.json', 'r') as file:
        fixation_events = json.load(file)

elif task == 'free_view':

    with open('../data/fixation_events_FV.json', 'r') as file:
        fixation_events = json.load(file)



if sigma < 50:
    if draw_details:
        heatmap_img = 'heatmap_small_sd_fx.png'
        masked_img = 'masked_small_sd_fx.png'

    else:
        heatmap_img = 'heatmap_small_sd.png'
        masked_img = 'masked_small_sd.png'

else:
    if draw_details:
        heatmap_img = 'heatmap_large_sd_fx.png'
        masked_img = 'masked_large_sd_fx.png'

    else:
        heatmap_img = 'heatmap_large_sd.png'
        masked_img = 'masked_large_sd.png'

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
    duration = (max_t - min_t)

    print(min_x, min_y, width, height)

    centroid = ((min_x + width/2), (min_y + height/2)) # eq to (minx + maxx) / 2

    if draw_details:
        #ax.add_patch(plt.Rectangle((min_x, min_y), width, height, fill=True, alpha=0.5, color='cyan', linewidth=1.5))

        circle = plt.Circle(centroid, 10, color='r', alpha=1)
        ax.add_artist(circle)

    return centroid, duration

grey_img_file = '../data/images_bordered/' + img_id + '.bmp'

img = pim.imread(grey_img_file)

image = Image.open(grey_img_file).convert('RGB')
#image = image.crop((206, 50, 206 + 1267, 50 + 950)) # crop to the size of the actual image on the screen

print(image.size)

plt.figure(0)

plt.axis('off')

ax = plt.gca()

#plt.imshow(img)
#imgplot = plt.imshow(image)

fixation_windows = fixation_events[ppn_id][img_id]

width = image.size[0]
height = image.size[1]

final_heatmap = np.zeros((height, width)) #1050, 1680 with borders reverse in np

durations = []
fixation_heatmaps = []
for f in fixation_windows:

    centroid, duration = get_fixation_window(f, ax, plt, draw_details)

    #generate a gaussian around the centroid over the whole image

    x = np.zeros((height, width))
    print(x.shape)

    # not the precise centroid, rather an approximation due to the discreteness of image pixels
    centroid = (round(centroid[1]), round(centroid[0])) #reverse in np

    x[centroid] = 1

    # https://github.com/durandtibo/heatmap/blob/master/heatmap/heat_map.py
    fixation_heatmap = ndimage.filters.gaussian_filter(x, sigma = sigma)
    print(fixation_heatmap.shape)

    durations.append(duration)

    fixation_heatmaps.append(fixation_heatmap)

    print(centroid)


# normalize the durations in the context of this image:
# RETHINK THE DURATIONS WHEN WE SKIP THINGS

durations = [d/sum(durations) for d in durations]

print(durations)

for fh in range(len(fixation_heatmaps)):

    if duration_weighted:
        final_heatmap += durations[fh]*fixation_heatmaps[fh]
    else:
        final_heatmap += fixation_heatmaps[fh]


max_value = np.max(final_heatmap)
min_value = np.min(final_heatmap)
normalized_heatmap = (final_heatmap - min_value) / (max_value - min_value)

print(normalized_heatmap.shape)

new_img = img.copy()

# mask the image given the heatmap
new_img[:,:,0] = new_img[:,:,0] * normalized_heatmap
new_img[:,:,1] = new_img[:,:,1] * normalized_heatmap
new_img[:,:,2] = new_img[:,:,2] * normalized_heatmap

plt.imshow(new_img)

plt.savefig(masked_img) #this has blank spaces around

# CROP the masked image and save it so that we can obtain its features
masked_image = Image.fromarray(new_img, 'RGB')
masked_image = masked_image.crop((206, 50, 206 + 1267, 50 + 950)) # crop to the size of the actual image on the screen
masked_image.save('masked_' + ppn_id + '_' + img_id + '.png')

plt.figure(1)

plt.axis('off')

ax = plt.gca()

plt.imshow(image) #this has gray borders around

normalized_heatmap = 255*normalized_heatmap

#https://github.com/esdalmaijer/PyGazeAnalyser/blob/master/pygazeanalyser/gazeplotter.py
lowbound = np.mean(normalized_heatmap[normalized_heatmap>0])
print('lowbound', lowbound)
normalized_heatmap[normalized_heatmap<lowbound] = np.NaN

plt.imshow(normalized_heatmap, alpha=0.9, cmap = 'plasma') #this has blank spaces around

#cmaps 'viridis', 'plasma', 'inferno', 'magma', 'nipy_spectral'
#plt.colorbar()

for f in fixation_windows:

    centroid, duration = get_fixation_window(f, ax, plt, draw_details)

plt.savefig(heatmap_img) #this has blank spaces around
plt.show()