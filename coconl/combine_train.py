import json

# combining 3 subsets of train

translated_train_coco_file1 = "captions_train2017_NL1.json"
translated_train_coco_file2 = "captions_train2017_NL2.json"
translated_train_coco_file3 = "captions_train2017_NL3.json"

caption_img_ids1 = 'train_cap_img_ids1.json'
caption_img_ids2 = 'train_cap_img_ids2.json'
caption_img_ids3 = 'train_cap_img_ids3.json'

combined_train_coco = 'captions_train2017_NL_full.json'
combined_image_ids = 'train_cap_img_ids_full.json'

# captions

combined_captions = []

with open(translated_train_coco_file1, 'r') as f:
    subset = json.load(f)

for s in subset:
    combined_captions.append(s)

with open(translated_train_coco_file2, 'r') as f:
    subset = json.load(f)

for s in subset:
    combined_captions.append(s)

with open(translated_train_coco_file3, 'r') as f:
    subset = json.load(f)

for s in subset:
    combined_captions.append(s)

print(len(combined_captions))

with open(combined_train_coco, 'w') as f:
    json.dump(combined_captions, f)

# image ids

combined_ids = []

with open(caption_img_ids1, 'r') as f:
    subset = json.load(f)

for s in subset:
    combined_ids.append(s)

with open(caption_img_ids2, 'r') as f:
    subset = json.load(f)

for s in subset:
    combined_ids.append(s)

with open(caption_img_ids3, 'r') as f:
    subset = json.load(f)

for s in subset:
    combined_ids.append(s)

print(len(combined_ids))

with open(combined_image_ids, 'w') as f:
    json.dump(combined_ids, f)