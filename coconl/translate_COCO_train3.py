# pip install --upgrade google-cloud-translate
# authenticate

# Imports the Google Cloud client library
from google.cloud import translate
import json

# Instantiates a client
translate_client = translate.Client()

# The source language
source = 'en'
# The target language
target = 'nl'

train_coco_file = "annotations_trainval2017/annotations/captions_train2017.json"

translated_train_coco_file = "captions_train2017_NL3.json"

with open(train_coco_file, 'r') as f:
    train_captions = json.load(f)

captions = []
caption_img_ids = []

train_captions_subset = train_captions['annotations'][534469:]

for a in range(len(train_captions_subset)):
    caption_item = train_captions_subset[a]
    captions.append(caption_item['caption'])
    caption_img_ids.append(caption_item['image_id'])

with open('train_cap_img_ids3.json', 'w') as f:
    json.dump(caption_img_ids, f)

translated_train = []

cap_count = len(captions)

i = 0
chunk = 0
call_flag = True

while call_flag:

    chunk += 1
    print(chunk, i)

    if (i+128) < cap_count:
        translations = translate_client.translate(captions[i:i+128], source_language=source, target_language=target)
        i = i + 128

    else:
        translations = translate_client.translate(captions[i:], source_language=source, target_language=target)
        call_flag = False

    for t in range(len(translations)):
        translated_train.append(translations[t]['translatedText'])
        print(translations[t]['translatedText'])


with open(translated_train_coco_file, 'w') as f:
    json.dump(translated_train, f)
