# Generating Image Descriptions via Sequential Cross-Modal Alignment Guided by Human Gaze

Repository for the EMNLP 2020 paper ['Generating Image Descriptions via Sequential Cross-Modal Alignment Guided by Human Gaze'](https://www.aclweb.org/anthology/2020.emnlp-main.377/) by Ece Takmaz, Sandro Pezzelle, Lisa Beinborn, Raquel Fernández.

This subdirectory contains the model definitions and scripts for training and evaluating these models, building on the PyTorch implementation of the Bottom-Up and Top-Down Attention for Image Captioning model by Anderson et al. (2018) at [https://github.com/poojahira/image-captioning-bottom-up-top-down](https://github.com/poojahira/image-captioning-bottom-up-top-down).

I will list the requirements later and share the large data files as well as trained model files.

Model names:
- model_image: Used in pretraining the model on translated MS COCO
- model_pret_image: Identical to model_image (used in finetuning on DIDEC)
- model_pret_gaze_SM: **NO-GAZE** - The model that takes features of an image masked with an aggregated saliency maps
- model_pret_gaze_RNN: **GAZE-SEQ** - The model that takes features of an image masked with an sequential saliency maps
- model_pret_gaze_2RNN: **GAZE-2SEQ** - The model that has an additional LSTM processing gaze-conditioned image features

To pretrain the model on translated MS COCO:

``python train_image_COCONL.py -learning_rate 0.0001 -batch_size 128 -dropout 0.0 -shuffle_flag -emb_dim 1024 -epochs 100 -data_folder final_dataset -metric cider``

The selected pretrained model is ['model_COCONLimage_cider_2019-10-31-13-53-9.pkl'](https://uva.data.surfsara.nl/index.php/s/Lw4SMBHYyaPe37q). To further finetune on this model, an example command would be:

``python train_pretrained_2RNN_NEWMETR.py -seed 42 -checkpoint 'model_COCONLimage_cider_2019-10-31-13-53-9.pkl' -learning_rate 0.0001 -batch_size 64 -dropout 0.0 -shuffle_flag -emb_dim 1024 -epochs 100 -data_folder final_dataset -metric ours``

-metric ours indicate that we use our proposed metric, Semantic and Sequential Distance, SSD. Its implementation can be found at description_generation/utils/Metric_New.py

To evaluate a trained model:

``python eval_pret_2rnn_NWMT.py -split 'test' -checkpoint [model_file_name]``

-split can be 'test' or 'val'.

For pretraining:
CAPLENS, CAPTIONS, FEATS, REFS, ids files with _coconl extension for all 3 splits.

For finetuning:
CAPLENS, CAPTIONS, FEATS, REFS, ids files with _didec extension for all 3 splits.

ALGFEATS (features of image masked with sequential masks aligned with uttered words)

MASKFEATS (features of image masked with a single aggregated mask)

Building on the PyTorch implementation of bottom-up and top-down attention for image captioning model at [https://github.com/poojahira/image-captioning-bottom-up-top-down](https://github.com/poojahira/image-captioning-bottom-up-top-down).

Bottom-up features downloaded from [https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip](https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip)

- train36.hdf5
- val36.hdf5
- train36_imgid2idx.pkl
- val36_imgid2idx.pkl


---
Emiel van Miltenburg, Ákos Kádár, Ruud Koolen, and Emiel Krahmer. 2018. DIDEC: The Dutch Image Description and Eye-tracking Corpus. In Proceedings of the 27th International Conference on Computational Linguistics (COLING), pages 3658–3669. Association for Computational Linguistics

Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. 2018. Bottom-Up and Top-Down Attention for Image Captioning and VQA. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6077–6086.
