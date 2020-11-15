# Generating Image Descriptions via Sequential Cross-Modal Alignment Guided by Human Gaze

Repository for the EMNLP 2020 paper ['Generating Image Descriptions via Sequential Cross-Modal Alignment Guided by Human Gaze'](https://www.aclweb.org/anthology/2020.emnlp-main.377/) by Ece Takmaz, Sandro Pezzelle, Lisa Beinborn, Raquel Fernández.

For any questions regarding the contents of this repository, please contact Ece Takmaz at <ece.takmaz@uva.nl>.

\~Currently under construction\~

You can find more details in the README files of each subdirectory.

For more details on the models (architectures, training and evaluation) look at **description_generation**. 

For the preprocessing steps we performed on the DIDEC dataset, take a look at **data_processing** (processing fixations, masking images, audio-text alignment, fixation window-text alignment, creating the final dataset for the models). 

The code for machine-translating the MS COCO dataset into Dutch is under **coconl**, along with the resulting translations.

In **scanpath_analysis**, we provide the code and data for the cross-modal correlation analysis. 
 
