import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.image as pim

with open('../data/fixation_events_DS.json', 'r') as file: #DESCRIPTIONS
    fixations_dict = json.load(file)

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


aligned_fix_dict = dict()

count_g = 0

for p_no in fixations_dict:

    pp_alignments = dict()

    for image_id in fixations_dict[p_no]:

        print(p_no, image_id)

        # getting the fixation info
        # if they are aligned

        alignedFlag= isAligned(p_no, image_id, sets)

        if alignedFlag:

            count_g += 1

            if count_g % 100 == 0:
                print('count_g', count_g)

            fixation_windows = fixations_dict[p_no][image_id]

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
                            combined_sil_interval = (temp_sil_token, temp_sil_interval_begin, temp_sil_interval_end)

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
                            combined_sil_interval = (temp_sil_token, temp_sil_interval_begin, temp_sil_interval_end)

                            new_alignment_map.append(combined_sil_interval)


                    else:
                        combined_sil_interval = (temp_sil_token, temp_sil_interval_begin, temp_sil_interval_end)

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

            fx_window_current = 0

            fx_temp_count = 0

            fix4alignment_map = []

            history_of_fixs = dict()

            for ap in alignment_map:

                if ap[0] != '<sil>':

                    print(ap)

                    fix4ap = []

                    for w in fixation_windows[fx_window_current:]:

                        #current_timestamp, lx, ly, rx, ry = g
                        #print(w)
                        fix_begin = w[0][0]
                        fix_end = w[-1][0]

                        if float(ap[1]) > fix_begin:
                            fix4ap.append(w)
                            fx_temp_count += 1

                        else:
                            fx_window_current = fx_temp_count

                            break # should be attached to the next word

                        print(fx_temp_count, fix_begin, fix_end)

                    if len(fix4ap) == 0:
                        print('empty', ap)

                    fix4alignment_map.append((ap[0], fix4ap))

                    print()

            pp_alignments[image_id] = fix4alignment_map

    aligned_fix_dict[p_no] = pp_alignments

print(count_g)

img_per_participant = []

for ppn in fixations_dict:

    print(ppn, len(fixations_dict[ppn]))

    img_per_participant.append(len(set(fixations_dict[ppn].keys())))

print('participants', len(fixations_dict.keys()))
print('avg img per participant', sum(img_per_participant)/len(img_per_participant))

with open('../data/aligned_fixations_words.json', 'w') as file:
    json.dump(aligned_fix_dict, file)