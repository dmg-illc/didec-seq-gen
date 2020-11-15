import json
import os
import wave
import contextlib
import statistics
#prints some statistics about the audio recordings

audio_path = '../data/converted_wavs' # write the full path

#check only the aligned audio in the splits

with open('../data/split_train.json', 'r') as file:
    train_set = json.load(file)

with open('../data/split_val.json', 'r') as file:
    val_set = json.load(file)

with open('../data/split_test.json', 'r') as file:
    test_set = json.load(file)


train_durations = []
train_count = 0

val_durations = []
val_count = 0

test_durations = []
test_count = 0


for subdirs, dirs, files in os.walk(audio_path):

    for f in files:
        #print(f)
        fname = os.path.join(audio_path, f)

        f_split = f.split('.')[0].split('_')

        audio_ppn = f_split[2][3:]
        audio_img = f_split[3]

        with contextlib.closing(wave.open(fname,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            #print(frames, rate, duration)


            if audio_img in train_set and audio_ppn in train_set[audio_img]:

                train_durations.append(duration)
                train_count += 1

            elif audio_img in val_set and audio_ppn in val_set[audio_img]:

                val_durations.append(duration)
                val_count += 1

            elif audio_img in test_set and audio_ppn in test_set[audio_img]:

                test_durations.append(duration)
                test_count += 1

print(train_count, val_count, test_count)

all_durations = {'train': train_durations, 'val': val_durations, 'test': test_durations}

for split in all_durations:

    durations = all_durations[split]
    print(split, min(durations), max(durations), statistics.mean(durations), statistics.pstdev(durations), sum(durations)/len(durations))

#3658 444 446
#train 1.4639375 35.7365625 10.299569676052487 4.185304627469579 10.299569676052496
#val 2.9268125 27.534125 10.531599802927929 4.326463393304341 10.53159980292793
#test 2.2215 32.2100625 10.197091087443946 4.180856804072588 10.197091087443951