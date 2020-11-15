# Generating Image Descriptions via Sequential Cross-Modal Alignment Guided by Human Gaze

Repository for the EMNLP 2020 paper ['Generating Image Descriptions via Sequential Cross-Modal Alignment Guided by Human Gaze'](https://www.aclweb.org/anthology/2020.emnlp-main.377/) by Ece Takmaz, Sandro Pezzelle, Lisa Beinborn, Raquel Fernández.

This subdirectory contains the code we implemented to process the DIDEC data: time-alignment of descriptions in textual form with the audio files, extracting fixation maps from the gaze data (sequential and aggregated), and the alignment of words with the extracted fixation windows.


**List of processing steps implemented in this folder:**
- Tokenizing the descriptions in the description view portion of DIDEC
- Creating grammar files for CMU Sphinx for the time alignment of audio and text (captions in the grammar files are the most processed versions of the captions)
- Converting the DIDEC audio files to the suitable format for CMU Sphinx
- Obtaining IPA countarparts of out-of-vocabulary words and mapping them to CMU Sphinx format
- Generating CMU Sphinx alignment commands
- Outcomes of audio-text alignment (each word with its timestamp + silent intervals)
- Parsing the fixation data from DIDEC, processing it to obtain fixation centroids from raw data
- Generating aggregated maps combining gaze info from multiple participants
- Generating sequential maps aligning timestamps of words with fixation windows
- Masking the images with the generated maps and obtaining the features of masked images
- Final dataset for COCONL: Creating the vocabulary and train-test-val files containing captions, caption lengths, references and masked image features (also check description_generation for the creation of the final dataset for finetuning)

I may share these data files later:
- **dict_gaze.json** (raw gaze data from DIDEC, parse_gaze_data.py)
- **fixation_events_DS.json** (processed fixation data, get_descr_fix.py)
- **aligned_fixations_words.json** (utterance-fixation alignment, )

In the data_processing/data folder, we put gaze_data (all participants under this folder), images and images_bordered from [DIDEC](https://didec.uvt.nl/pages/download.html). 
Other files stored in this folder: processed captions, grammar files, alignments and converted audio files. 

Bottom-up features downloaded from [https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip](https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip)

- train36.hdf5
- val36.hdf5
- train36_imgid2idx.pkl
- val36_imgid2idx.pkl

split_*.json (Visual Genome IDs of the selected images per split)

generate_COCONL_caps_refs_feats.py for the final data of pretraining

*Check description_generation/final_dataset for the creation of the final dataset for finetuning*

---
Emiel van Miltenburg, Ákos Kádár, Ruud Koolen, and Emiel Krahmer. 2018. DIDEC: The Dutch Image Description and Eye-tracking Corpus. In Proceedings of the 27th International Conference on Computational Linguistics (COLING), pages 3658–3669. Association for Computational Linguistics

Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. 2018. Bottom-up and top-down attention for image captioning and VQA. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6077–6086.