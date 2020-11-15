import json
import os
from utils.UtteranceTokenizer import UtteranceTokenizer
from collections import defaultdict

def process_gaze(gaze_file_name, all_annotations, tokenizer, dict_caption_audio, gaze_file_count, digit_list):

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

    for g in gazedata:
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
            # if l_por_x != r_por_x:
            #     print('xpos', timestamp)
            #
            # if l_por_y != r_por_y:
            #     print('lpos', timestamp)
            #
            # if r_event != l_event:
            #     print('event', timestamp)
            #
            #     #blinks and no event sometimes
            #
            # if int(trial_no) > 1:
            #     print('trial', timestamp)


            timestamps.append(timestamp)
            r_xs.append(r_por_x)
            r_ys.append(r_por_y)
            l_xs.append(l_por_x)
            l_ys.append(l_por_y)

            # ADD EVENT TYPES


        else:
            # message line
            img_no = g_split[5]
            image_id = img_no.split('.')[0]

            if image_id == 'UE-keypress':
                #print('MESSAGE', image_id)  # WARNING UE-KEYPRESS message, removed them
                pass
            else:
                for a in all_annotations:
                    if a['participant'] == p_no and a['image'] == image_id:
                        mp3_file_name = a['filename']

                        raw_caption = a['raw_description']
                        normalized_caption = a['normalized_description']

                        break

                #WARNING THERE WERE annotations with no proper raw or normalized captions, skipping them
                #already skipping ones without eye-data or mp3

                # {'713274'}
                # 35
                # {'1159675'}
                # 108
                # {'713640'}
                # 3
                # {'712977'}
                # 24
                #
                if normalized_caption not in ['', '-', '- ']:
                    mp3_folder_name = '../data/gaze_data/pp' + p_no + '/mp3'

                    mp3_file_path = os.path.join(mp3_folder_name, mp3_file_name)

                    #print(raw_caption, normalized_caption)
                    #print(mp3_file_path)

                    caption_file = '../data/captions/raw_caption_' + p_no + '_' + image_id + '.txt'
                    #write the words one by one

                    with open(caption_file, 'w') as file:
                        gaze_file_count +=1
                        tokenized_utterance = tokenizer.tokenize_utterance(utterance=raw_caption, method='nltk',
                                                                           lowercase=True)

                        #includes punctuation
                        for i in range(len(tokenized_utterance)):
                            #<corr>, <pause>, <rep>, <uh>

                            word = tokenized_utterance[i]

                            toWrite = False

                            if word == '<uh>':
                                #replace <uh> with the word uh
                                toWrite = True
                                word = 'uh'

                            elif word not in ['<rep>', '<corr>', '<pause>']:
                                #skip these and write the actual words
                                # maybe for pause write uh

                                has_numeric = False

                                for d in digit_list:
                                    if d in word:
                                        has_numeric = True
                                        break

                                if has_numeric:

                                    #print(mp3_file_name)
                                    # MAP TO ALPHABETICAL VERSION
                                    pass

                                elif len(word) == 1 and not word.isalpha():
                                    # THINGS LIKE comma might indicate some filler utterances
                                    #skip stand-alone punctuation marks (instances such as auto's remain, as well as single letter words)
                                    #print(word, mp3_file_name, tokenized_utterance)
                                    # this could also pass single digit numbers, check them before this clause

                                    pass

                                else:
                                    #all other words and numbers
                                    # THERE ARE NUMBERS WRITTEN NUMERICALLY, HOW DOES cmuspinhxm WORK?
                                    toWrite = True

                            if toWrite:
                                if word == '<?>':
                                    #print(mp3_file_name, tokenized_utterance) # THESE ARE UNINTELLIGIBLE WORDS

                                    pass
                                    # problem is in the ones that have ? in the middle, so, fewer of them actually are problematic

                                    #normalized ones just remove the <?>, so that's what I do as well
                                    #I skip this token and don't write it
                                    #print(normalized_caption)


                                else:
                                    file.write(word)
                                    if i < len(tokenized_utterance)-1:
                                        file.write('\n')


                    dict_caption_audio[caption_file] = mp3_file_path


    trial_dict = {'timestamps':timestamps, 'rxs':r_xs, 'rys':r_ys, 'lxs':l_xs, 'lys':l_ys, }

    return p_no, image_id, trial_dict, gaze_file_count


with open('../data/annotations_all_participants.json', 'r') as file:
    all_annot = json.load(file)


if not os.path.isdir('../data/captions'):
    os.mkdir('../data/captions')

tokenizer = UtteranceTokenizer('nl')

gazedir = '../data/gaze_data'
gaze_file_count = 0

dict_caption_audio = defaultdict(str)
dict_gaze = defaultdict()

# gaze data for some trials are missing, but we have the speech and annotations
# instead I'm looking at the gaze data and retrieve the captions and audio for the existing gaze data
# pp p_no eye_len mp3_len
# pp 43 91 103
# pp 47 101 102
# pp 49 102 103


digit_list = []

for i in range(10):
    digit_list.append(str(i))

number2alpha = dict()
number2alpha['3'] = 'drie'
number2alpha['7'] = 'zeven'
number2alpha['08'] = 'acht'
number2alpha['9'] = 'negen'
number2alpha['11'] = 'elf'
number2alpha['15'] = 'vijftien'
number2alpha['16'] = 'zestien'
number2alpha['21'] = 'eenentwintig' #wrong caption en een brunette...
number2alpha['39'] = 'negenendertig'
number2alpha['43'] = 'drieenveertig'
number2alpha['46'] = 'zeseenveertig'
number2alpha['49'] = 'negenenveertig' #also fourty nine
number2alpha['50'] = 'vijftig'
number2alpha['70'] = 'zeventig'
number2alpha['117'] = 'honderdzeventien' #also een een zeven
number2alpha['360'] = 'driezestig'
number2alpha['521'] = 'vijfhonderdeenentwintig'
number2alpha['5143'] = 'vijf een vier drie'
number2alpha['x50'] = 'x vijftig'
number2alpha['500w'] = 'five hundred w' #double u, not the dutch w
number2alpha['2011-7'] = 'twee elf zeven hondred acht' #twee elf zeven hondred acht vijf uur drie en dertig CAPTION IS NOT GOOD
number2alpha['5:33'] = 'vijf uur drieendertig ' #twee elf zeven hondred acht vijf uur drie en dertig CAPTION IS NOT GOOD
#http://cocodataset.org/#explore?id=45337  27/08/2011 5:33 PM

for root, dirs, files in os.walk(gazedir):

    if 'log+raw' not in root: #only look at eye and mp3 folders
        for f in files:

            if 'txt' in f:
                #process gaze data

                file_name = os.path.join(root, f)

                p_no, image_id, trial_dict, gaze_file_count = process_gaze(file_name, all_annot, tokenizer, dict_caption_audio,gaze_file_count, digit_list)

                if p_no in dict_gaze:
                    dict_gaze[p_no].update({image_id: trial_dict})

                else:
                    dict_gaze[p_no] = {image_id: trial_dict}

    # if 'eye' in root:
    #     print('e',len(files))
    #
    # if 'mp3' in root:
    #     print('m',len(files))


print(gaze_file_count) #instances with proper captions and audio and gaze

with open('../data/dict_caption_audio_mp3.json', 'w') as file:
    json.dump(dict_caption_audio, file)

with open('../data/dict_gaze.json', 'w') as file:
    json.dump(dict_gaze, file)
