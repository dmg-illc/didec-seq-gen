# Generating Image Descriptions via Sequential Cross-Modal Alignment Guided by Human Gaze

Repository for the EMNLP 2020 paper ['Generating Image Descriptions via Sequential Cross-Modal Alignment Guided by Human Gaze'](https://www.aclweb.org/anthology/2020.emnlp-main.377/) by Ece Takmaz, Sandro Pezzelle, Lisa Beinborn, Raquel Fernández.

This subdirectory contains the code for **machine-translating MS COCO into Dutch** using the Google Translate API and the resulting dataset that we then preprocessed to use in the pretraining of our models.

Download the **annotations_trainval2017** folder from MS COCO from [http://images.cocodataset.org/annotations/annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

**translate_COCO_train.py**, **translate_COCO_train2.py**, **translate_COCO_train3.py** and **translate_COCO_val.py** make calls to the Google Translate API to obtain the translated captions. I left out the authentication part, as you need to use your own authentication info if you would like to run them. 

**captions_train2017_NL_full.json** and **captions_val2017_NL.json** contain the translated captions for the train and val splits of MS COCO.

Here, we also create the combined vocabulary of DIDEC and translated MS COCO (**WORDMAP_union.json**).

For the pretraining part, you can find the model definition, training and evaluation scripts in the description_generation folder, look for model_image.py, train_image_COCONL.py and eval_image_COCONL.py.
The dataset files used in pretraining are created in generate_COCONL_caps_refs_feats.py under data_processing/utils.
