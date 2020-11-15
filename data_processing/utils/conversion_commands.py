import json
import os

#creates the shell script for the commands to convert mp3s to the correct format
#so that CMUSphinx can work (default params)
#for all the captions and the corresponding audio files

#THIS INCLUDES MP3s, json file generated in parse_gaze_data.py
with open('../data/dict_caption_audio_mp3.json', 'r') as file:
    dict_caption_audio = json.load(file) #caption path -> audio path

main_folder = 'data_processing'  # put the full path here

converted_audio_path = '../data/converted_wavs'
if not os.path.isdir(converted_audio_path):
    os.mkdir(converted_audio_path)

#required format
acodec = 'pcm_s16le'
channel_count = '1'
frequency = '16000'
file_type = '.wav'

count = 0

new_dict_caption_audio = dict()

with open('convert_commands.sh', 'w') as file:

    for caption_path in dict_caption_audio:

        audio_path = dict_caption_audio[caption_path]

        record_name = audio_path.split('/')[5]

        converted_record_name = record_name.split('.')[0] + file_type

        converted_file_path = os.path.join(converted_audio_path, converted_record_name)

        command_string = 'ffmpeg -i ' + audio_path + ' -acodec ' + acodec + \
                         ' -ac ' + channel_count + ' -ar ' + frequency + ' ' + converted_file_path

        print(command_string)

        file.write(command_string)
        file.write('\n')
        count += 1

        relative_wav_path =  '../'

        append = False

        split_file_path = converted_file_path.split('/')

        for w in range(len(split_file_path)):

            path_part = split_file_path[w]

            if append:
                relative_wav_path += path_part

            if path_part == 'data':
                append = True
                relative_wav_path += path_part

            if append and w < len(split_file_path)-1:
                relative_wav_path += '/'

        new_dict_caption_audio[caption_path] = relative_wav_path

    print(count)

#update the json file
#THIS INCLUDES WAV FILES, to be used in alignment_commands_cmu.py
with open('../data/dict_caption_audio_wav.json', 'w') as file:
    json.dump(new_dict_caption_audio, file) # caption path -> new audio path wavs