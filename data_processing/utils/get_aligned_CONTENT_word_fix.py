import json
import spacy

# MANY manual specific modifications because spacy does tokenization in pos-tagging in a different way
# causing problems with ' - and so on
#

nlp = spacy.load("nl_core_news_sm")

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

content_captions_dict = dict()

for p_no in fixations_dict:

    pp_alignments = dict()
    pp_captions = dict()

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

            # GETTING THE CAPTION IN THE GRAMMARS, THEY ARE THE MOST PROCESSED ONES
            grammar_file = '../data/grammars/raw_caption_' + str(p_no) + '_' + str(image_id) + '.jsgf'

            with open(grammar_file, 'r') as file:
                grammar_lines = file.readlines()

            content_caption = []
            content_indices = []

            ind = 0

            for line in grammar_lines:
                if '=' in line:
                    split_line = line.split('=')

                    caption = split_line[1].split(';')[0]

                    # checked lowercase...
                    # checked inital space...

                    # if not caption.islower(): # islower true if none is upper
                    #     print('UPPERCASE', caption)
                    #
                    # if not caption[0] == ' ':
                    #     print('NONSPACE', caption) # not tokenizing the caption, but there is a space at the beginning, from the grammar

                    caption = caption[1:]  # skip the space at the beginning

                    doc = nlp(caption)

                    print(doc.text)

                    skip_token_indices = []

                    for t in range(len(doc)):

                        if t not in skip_token_indices:
                            token = doc[t]

                            if token.pos_ == 'NOUN' or token.pos_ == 'ADJ' or token.pos_ == 'VERB':

                                token_text = token.text

                                if (t+1) < len(doc) and doc[t+1].text == '\'s': # SPACY TOKENIZES 's PLURAL
                                    token_text += '\'s'

                                    skip_token_indices.append(t + 1)

                                elif (t+1) < len(doc) and doc[t+1].text == '\'': # SPACY TOKENIZES in-word '

                                    token_text += '-'
                                    token_text += doc[t + 2].text
                                    skip_token_indices.append(t + 1)
                                    skip_token_indices.append(t + 2)

                                    assert (t+2) < (len(doc)-1)

                                elif (t+1) < len(doc) and doc[t+1].text == '-': # SPACY TOKENIZES in-word -

                                    token_text += '-'
                                    token_text += doc[t+2].text

                                    assert (t+2) < len(doc)

                                    skip_token_indices.append(t + 1)
                                    skip_token_indices.append(t + 2)

                                    if (t+3) < len(doc) and doc[t+3].text == '-': # second - in the word such as : kant-en-klare

                                        token_text  += '-'
                                        token_text += doc[t + 4].text

                                        skip_token_indices.append(t + 3)
                                        skip_token_indices.append(t + 4)

                                        assert (t+4) < (len(doc) - 1)

                                    elif (t+3) < len(doc) and doc[t+3].text == '\'s': # post-it's
                                        #een rondvormige desk met drie computers erop en
                                        #een witte laptop met roze hoesje en uh post-it's op de computers

                                        token_text  += '\'s'
                                        skip_token_indices.append(t + 3)

                                        assert (t + 3) < len(doc)

                                content_caption.append(token_text)
                                content_indices.append(ind)

                            elif token.pos_ != 'NOUN' or token.pos_ != 'ADJ' or token.pos_ != 'VERB':

                                token_text = token.text

                                if (t + 1) < len(doc) and doc[t + 1].text == '-':  # SPACY TOKENIZES in-word -

                                    if (t + 2) < len(doc) and doc[t + 2].pos_ in ['NOUN', 'ADJ', 'VERB']:

                                        token_text += '-'
                                        token_text += doc[t + 2].text

                                        assert (t + 2) < len(doc)

                                        skip_token_indices.append(t + 1)
                                        skip_token_indices.append(t + 2)

                                        content_caption.append(token_text)
                                        content_indices.append(ind)
                            ind += 1

            for l in range(len(alignment_lines)):

                if l == 0:

                    print(alignment_lines[l])
                    # whole raw caption

                else:
                    # alignment of tokens
                    alignment_per_item = alignment_lines[l]

                    split_alignment = alignment_per_item.split()

                    #print(split_alignment)

                    token_al = split_alignment[0]

                    # (2) for some items, alternative pronunciation
                    if '(2)' in token_al:
                        token_al = token_al.split('(')[0] #there are alternative phonetic transcriptions for one words
                        #such as 117 - one one seven or one hundred seventeen

                    token_begin = split_alignment[1]
                    token_end = split_alignment[2]
                    # token_confidence = split_alignment[3] #not used prob

                    alignment_map.append((token_al, token_begin, token_end))

            begin_index = 0
            content_indices_in_aligned = []

            for cw in content_caption:

                for a_i in range(begin_index, len(alignment_map)):

                    if alignment_map[a_i][0] == cw:
                        content_indices_in_aligned.append(a_i)

                        begin_index = a_i + 1
                        break

            print(alignment_map)
            print(content_caption)
            print(content_indices_in_aligned)

            assert len(content_caption) == len(content_indices_in_aligned)

            # checking for the length (repeated words are still considered separately)

            # EXAMPLE AssertionError BECAUSE OF 'S
            # een
            # regenachtige
            # dag
            # met
            # auto
            # 's die rijden bij een stoplicht
            # een
            # regenachtige
            # dag
            # met
            # auto
            # 's die rijden bij een stoplicht
            #
            # ['regenachtige', 'dag', 'auto', 'rijden', 'stoplicht']
            # [2, 3, 8, 12]


            # COMBINE CONSECUTIVE <SIL> intervals
            # all token repetitions
            # {'nee', 'twee', 'aan', 'op', 'een', 'o', 'die', 'waar', 'je', 'oude',
            # 'stadsmensen', 'met', '<sil>', 'de', 'het', 'dat', 'wat', 'haar'}


            # HERE ALSO COMBINING NON-CONTENT WORDS AS IF THEY ARE SILENT TOKENS

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

                    elif  i not in content_indices_in_aligned:

                        # TREAT IT AS a new silent interval

                        temp_sil_token = "<sil>" # alignment_map[i][0]  # <sil>
                        temp_sil_interval_begin = alignment_map[i][1]
                        temp_sil_interval_end = alignment_map[i][2]

                        silFlag = True

                        # also add if there is an un-added interval at the end:

                        if i == len(alignment_map) - 1:
                            combined_sil_interval = (temp_sil_token, temp_sil_interval_begin, temp_sil_interval_end)

                            new_alignment_map.append(combined_sil_interval)

                    elif i in content_indices_in_aligned:
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

                    elif i not in content_indices_in_aligned:

                        # TREAT IT AS A silent token

                        # consecutive silence
                        # only update end timestamp

                        temp_sil_interval_end = alignment_map[i][2]

                        # also add if there is an un-added interval at the end:

                        if i == len(alignment_map) - 1:
                            combined_sil_interval = (temp_sil_token, temp_sil_interval_begin, temp_sil_interval_end)

                            new_alignment_map.append(combined_sil_interval)

                    elif i in content_indices_in_aligned:
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

            pp_captions[image_id] = content_caption

    aligned_fix_dict[p_no] = pp_alignments
    content_captions_dict[p_no] = pp_captions

print(count_g)

with open('../data/CONTENT_captions.json', 'w') as file:
    json.dump(content_captions_dict, file)

with open('../data/aligned_fixations_CONTENT_words.json', 'w') as file:
    json.dump(aligned_fix_dict, file)