import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.image as pim

draw = False

sequential = True #if sequential = True, shows the scan path of gaze, otherwise a heatmap
fast = True #displays the scan path at a faster rate

with open('../data/split_train.json', 'r') as file:
    train_set = json.load(file)

with open('../data/split_val.json', 'r') as file:
    val_set = json.load(file)

with open('../data/split_test.json', 'r') as file:
    test_set = json.load(file)

sets = (train_set, val_set, test_set)

def isAligned(p_no, img_id, sets):

    # returns true if there is an alignment for this pair of participant and image in any of the splits

    aligned = False

    train_set, val_set, test_set = sets

    if (img_id in train_set and p_no in train_set[img_id]) \
        or (img_id in val_set and p_no in val_set[img_id]) \
        or (img_id in test_set and p_no in test_set[img_id]):
        aligned = True

    return aligned

def is_gaze_on_grey(x,y):
    '''
    GREY REGION
    x range
    0-205 1473-1679
    y range
    0-49 1000-1049
    '''

    #skip if on grey region, as well as other outside image coords (like negative values)

    # AS IN remove_out_of_bounds from salicon_code.py
    '''coords = [get_coords(entry) for entry in entries
                                if entry['L Event Info'] == 'Fixation'
                                and entry['R Event Info'] == 'Fixation']
    if remove_out_of_bounds:
        height, width = DISPLAY_SIZE
        coords = [(x,y) for x,y in coords if x < width and y < height]
    return coords'''

    greyFlag = False

    if x <= 205 or x >= 1473 or y <= 49 or y >= 1000:
        greyFlag = True

    return greyFlag

def draw_fixation_window(fixation, ax, plt):

    all_xs = []
    all_ys = []

    for gaze_item in fixation:

        # LEFT AND RIGHT POSITIONS ARE THE SAME
        #gaze_item = (timestamp,l_por_x, l_por_y, r_por_x, r_por_y)
        # 1,2 or 3,4

        all_xs.append(float(gaze_item[3]))
        all_ys.append(float(gaze_item[4]))

    min_x = min(all_xs)
    max_x = max(all_xs)
    min_y = min(all_ys)
    max_y = max(all_ys)

    width = max_x - min_x
    height = max_y - min_y
    # centroid is later rounded in the heatmap, so I am not adding 1 here

    ax.add_patch(plt.Rectangle((min_x, min_y), width, height, fill=True, alpha=0.5,
                                   color='cyan', linewidth=1.5))

    print(min_x, min_y, width, height)
    centroid = ((min_x + width / 2), (min_y + height / 2))

    return centroid

def process_gaze(gaze_file_name):

    with open(gaze_file_name, 'r') as file:
        gazedata = file.readlines()

    #parse to get participant id
    #../data/gaze_data/pp106/eye/Ruud_exp3_list1_v1_ppn106_008_Trial041 Samples.txt

    p_no = gaze_file_name.split()[0].split('_')[5][3:]
    #print(p_no)

    timestamps = []
    r_xs = []
    r_ys = []
    l_xs = []
    l_ys = []

    gaze_count = 0

    ''' separate L-R events, but the coords are the same. I decided to use only when both are FIX.
    as in DIDEC example code
    
    fixation_flag_R = False
    fixation_flag_L = False

    fixation_windows_R = []
    fixation_windows_L = []

    current_fixation_window_R = []
    current_fixation_window_L = []'''

    fixation_flag = False

    fixation_windows= []

    current_fixation_window = []

    line_count = 0

    first_timestamp = 0  # get this timestamp from the first line of the gaze info, subtract from the rest

    for g in gazedata:

        line_count += 1

        g_split = g.split()

        if g_split[0] == '##' or g_split[0] == 'Time':
           pass #log info or header
        elif g_split[1] != 'MSG': #if MSG, it's a message, without gaze data

            if len(g_split) == 14 :
                timestamp, stimulus_type, trial_no, l_por_x, l_por_y, r_por_x, r_por_y,\
                timing, pupil_confidence, l_plane, r_plane, l_event, r_event, stimulus_name = g_split

            else:
                timestamp, stimulus_type, trial_no, l_por_x, l_por_y, r_por_x, r_por_y, \
                timing, pupil_confidence, l_plane, r_plane, aux, l_event, r_event, stimulus_name = g_split #EXTRA AUX HERE

            gaze_count += 1

            if gaze_count == 1:
                first_timestamp = timestamp


            #if r_event != l_event:
            #    print('event', timestamp, r_event, l_event)

            # NORMALIZE TIMESTAMPS

            timestamp = (float(timestamp) - float(first_timestamp)) / 1000000

            #keeping track of the fixation windows

            gaze_item = (timestamp,l_por_x, l_por_y, r_por_x, r_por_y)
            # print(gaze_item, r_event, l_event, fixation_flag_R, fixation_flag_L)

            if l_por_x != r_por_x or l_por_y != r_por_y:
                print('NOT THE SAME') # NEVER PRINTED OUT! LEFT AND RIGHT COORDS THE SAME

            # CHECK IF there are blinks and no-event items
            # as well as gaze on grey border and skip them
            # only use when they are both fixation events (l and r)

            # blinks and no event sometimes -

            # print if gaze is out of image, there are cases of L-F R-F out-of-image

            #if is_gaze_on_grey(float(l_por_x), float(l_por_y)) or is_gaze_on_grey(float(r_por_x), float(r_por_y)):
            #    if l_event == 'Fixation' and r_event == 'Fixation':
            #        print('GAZE ON GREY', gaze_file_name, gaze_item, l_event, r_event)

            # PROCESSING EVENTS

            # both eyes at the same time

            # append only when l and r events are fixations

            '''if float(l_por_x) < 0 or float(l_por_y) < 0:
                print('NEGATIVE', gaze_item, l_event, r_event) # there exist L FIX R FIX events with negative values'''

            if r_event == 'Fixation' and l_event == 'Fixation' and fixation_flag is True:
                # continue adding items to the current fixation window
                # if on actual image

                if not is_gaze_on_grey(float(l_por_x), float(l_por_y)) and not is_gaze_on_grey(float(r_por_x), float(r_por_y)):

                    current_fixation_window.append(gaze_item)

            elif r_event == 'Fixation'and l_event == 'Fixation' and fixation_flag is False:

                # new fixation window
                fixation_flag = True
                assert len(current_fixation_window) == 0

                # add gaze to window if on actual image
                if not is_gaze_on_grey(float(l_por_x), float(l_por_y)) and not is_gaze_on_grey(float(r_por_x), float(r_por_y)):

                    current_fixation_window.append(gaze_item)

            elif (r_event != 'Fixation' or l_event != 'Fixation') and fixation_flag is True:

                # if at least one of the events is not a fixation, end the current fx
                # end the fixation window
                fixation_flag = False

                # since we are skipping out-of-image gaze, it is possible that some windows
                # only contain such gaze data. if the window len is 0, this could be the case

                if len(current_fixation_window) > 0:
                    fixation_windows.append(current_fixation_window)
                    current_fixation_window = []

            elif (r_event != 'Fixation' or l_event != 'Fixation') and fixation_flag is not True:

                # nothing to do here
                pass

            if line_count == len(gazedata) - 1:
                # if there is a current window of fixations, also add that at the end of the file

                if len(current_fixation_window) > 0:
                    fixation_windows.append(current_fixation_window)

            timestamps.append(timestamp)
            r_xs.append(r_por_x)
            r_ys.append(r_por_y)
            l_xs.append(l_por_x)
            l_ys.append(l_por_y)

        else:
            # message line
            img_no = g_split[5]
            if 'UE-keypress' not in img_no:
                image_id = img_no.split('.')[0]

            '''if 'UE-keypress' in img_no:
                print(g_split)

                ['14686055344', 'MSG', '1', '#', 'Message:', 'UE-keypress', 'B']
                ['25258347277', 'MSG', '1', '#', 'Message:', 'UE-keypress', 'LeftAlt']
                ['27061196395', 'MSG', '1', '#', 'Message:', 'UE-keypress', 'N']
            
            '''

    # has all raw gaze data (fix, sac, bl etc)
    trial_dict = {'timestamps':timestamps, 'rxs':r_xs, 'rys':r_ys, 'lxs':l_xs, 'lys':l_ys, }

    return p_no, image_id, trial_dict, fixation_windows


alignments_path = '../data/alignments'

gazedir = '../data/gaze_data'

dict_gaze_events = defaultdict()

# gaze data for some trials are missing, but we have the speech and annotations
# instead I'm looking at the gaze data and retrieve the captions and audio for the existing gaze data, as well as the ones with alignments
# pp p_no eye_len mp3_len
# pp 43 91 103
# pp 47 101 102
# pp 49 102 103

#left eye event might differ from right eye event
#even then, the x-y positions for L and R are the same

fixations_dict = dict()

count_g = 0
for root, dirs, files in os.walk(gazedir):

    if 'log+raw' not in root: #only look at eye and mp3 folders

        for f in files:

            if 'txt' in f:
                #process gaze data

                file_name = os.path.join(root, f)

                p_no, image_id, trial_dict, fixation_windows = process_gaze(file_name)

                print(p_no, image_id)

                # getting the fixation info
                # if they are aligned

                alignedFlag= isAligned(p_no, image_id, sets)

                if alignedFlag:

                    count_g += 1

                    if count_g % 100 == 0:
                        print(count_g)

                    if p_no in fixations_dict:
                        fixations_dict[p_no].update({image_id: fixation_windows})

                    else:
                        fixations_dict[p_no] = {image_id: fixation_windows}

                    # getting the alignment info

                    mapped_file = '../data/alignments/cmu_mapped_' + str(p_no) + '_' + str(image_id) + '.txt'

                    with open(mapped_file, 'r') as alg_f:
                        alignment_lines = alg_f.readlines()

                    alignment_map = []

                    for l in range(len(alignment_lines)):

                        if l == 0:

                            print(alignment_lines[l])
                            # whole raw caption

                        else:
                            # alignment of tokens
                            alignment_per_item = alignment_lines[l]

                            split_alignment = alignment_per_item.split()

                            #print(split_alignment)

                            token = split_alignment[0]

                            # (2) for some items, alternative pronunciation
                            if '(2)' in token:
                                token = token.split('(')[0] #there are alternative phonetic transcriptions for one words
                                #such as 117 - one one seven or one hundred seventeen

                            token_begin = split_alignment[1]
                            token_end = split_alignment[2]
                            # token_confidence = split_alignment[3] #not used prob

                            alignment_map.append((token, token_begin, token_end))

                    # COMBINE CONSECUTIVE <SIL> intervals
                    # all token repetitions
                    # {'nee', 'twee', 'aan', 'op', 'een', 'o', 'die', 'waar', 'je', 'oude',
                    # 'stadsmensen', 'met', '<sil>', 'de', 'het', 'dat', 'wat', 'haar'}

                    new_alignment_map = []

                    temp_sil_token = ''
                    temp_sil_interval_begin = ''
                    temp_sil_interval_end = ''

                    silFlag = False

                    for i in range(len(alignment_map)):

                        if not silFlag:

                            if alignment_map[i][0] == '<sil>':

                                #new silent interval

                                temp_sil_token = alignment_map[i][0] # <sil>
                                temp_sil_interval_begin = alignment_map[i][1]
                                temp_sil_interval_end = alignment_map[i][2]

                                silFlag = True

                                # also add if there is an un-added interval at the end:

                                if i == len(alignment_map) - 1:
                                    combined_sil_interval = [temp_sil_token, temp_sil_interval_begin, temp_sil_interval_end]

                                    new_alignment_map.append(combined_sil_interval)

                            else:
                                temp_sil_token = ''
                                temp_sil_interval_begin = ''
                                temp_sil_interval_end = ''

                                new_alignment_map.append(alignment_map[i])

                        elif silFlag:

                            if alignment_map[i][0] == '<sil>':
                                # consecutive silence
                                # only update end timestamp

                                temp_sil_interval_end = alignment_map[i][2]

                                # also add if there is an un-added interval at the end:

                                if i == len(alignment_map) - 1:
                                    combined_sil_interval = [temp_sil_token, temp_sil_interval_begin, temp_sil_interval_end]

                                    new_alignment_map.append(combined_sil_interval)


                            else:
                                combined_sil_interval = [temp_sil_token, temp_sil_interval_begin, temp_sil_interval_end]

                                new_alignment_map.append(combined_sil_interval)

                                new_alignment_map.append(alignment_map[i])

                                silFlag = False

                    #print(alignment_map)

                    alignment_map = new_alignment_map

                    print(alignment_map) #updated map with silent intervals combined

                    # NO EMPTY ALIGNMENTS anymore if len(alignment_map) > 0:
                    current_token = ''
                    current_token_index = 0
                    all_tokens = len(alignment_map)

                    fx_count = 0

                    for w in fixation_windows:

                        fx_count += 1

                        # I take a window and 
                        ''' this looks at individual gaze items: 
                        for g in w:

                            current_timestamp, lx, ly, rx, ry = g

                            current_token = alignment_map[current_token_index]

                            print(fx_count, current_timestamp, current_token[0], float(current_token[1]), float(current_token[2]))

                            if current_timestamp > float(current_token[1]) and current_timestamp <= float(current_token[2]):

                                pass


                            elif current_timestamp > float(current_token[2]):
                                # go to the next token
                                if current_token_index < len(alignment_map) - 1:
                                    current_token_index += 1

                                else:
                                    pass # no change of words at the end, more gaze than audio'''


                    if draw:

                        grey_img_file = '../data/images_bordered/' + image_id + '.bmp'

                        img = pim.imread(grey_img_file)

                        plt.figure(0)

                        ax = plt.gca()

                        imgplot = plt.imshow(img)


                        for f in fixation_windows:

                            draw_fixation_window(f, ax, plt)

                        plt.show()


print(count_g)

img_per_participant = []

for ppn in fixations_dict:

    print(ppn, len(fixations_dict[ppn]))

    img_per_participant.append(len(set(fixations_dict[ppn].keys())))

print('participants', len(fixations_dict.keys()))
print('avg img per participant', sum(img_per_participant)/len(img_per_participant))

with open('../data/fixation_events_DS.json', 'w') as file: #DESCRIPTIONS
    json.dump(fixations_dict, file)