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

val_coco_file = "annotations_trainval2017/annotations/captions_val2017.json"

translated_val_coco_file = "captions_val2017_NL.json"

with open(val_coco_file, 'r') as f:
    val_captions = json.load(f)

captions = []
caption_img_ids = []

for a in range(len(val_captions['annotations'])):
    caption_item = val_captions['annotations'][a]
    captions.append(caption_item['caption'])
    caption_img_ids.append(caption_item['image_id'])

with open('val_cap_img_ids.json', 'w') as f:
    json.dump(caption_img_ids, f)

translated_val = []

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
        translated_val.append(translations[t]['translatedText'])
        print(translations[t]['translatedText'])


with open(translated_val_coco_file, 'w') as f:
    json.dump(translated_val, f)
