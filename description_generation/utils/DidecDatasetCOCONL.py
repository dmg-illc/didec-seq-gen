import os
import json
import torch
from collections import defaultdict
from torch.utils.data import Dataset
import h5py

class DidecDatasetCOCONL(Dataset):

    def __init__(self, data_folder, data_name, split, model_type):

        self.max_ref = 7 #max ref per img in all the splits

        # {5, 6, 7} tr
        # {5, 6} v
        # {5, 6, 7} ts

        self.model_type = model_type

        self.data_folder = data_folder
        self.split = split
        assert self.split in {'train', 'val', 'test'}

        #coco nl

        # maxlen tr,val,ts
        # 54
        # 45
        # 40

        #for didec: 55,44,42

        if self.split == 'train':
            self.max_cap_len = 54  # max caption length in the train split
        elif self.split == 'val':
            self.max_cap_len = 45  # max caption length in the val split
        elif self.split == 'test':
            self.max_cap_len = 40  # max caption length in the test split


        self.data_name = data_name

        # Open hdf5 filea where image features are stored
        self.train_hf = h5py.File('final_dataset/train36.hdf5', 'r')
        self.train_features = self.train_hf['image_features']

        self.val_hf = h5py.File('final_dataset/val36.hdf5', 'r')
        self.val_features = self.val_hf['image_features']

        split_name = self.split

        # Load encoded captions
        with open(os.path.join(data_folder, split_name + '_CAPTIONS_' + self.data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths
        with open(os.path.join(data_folder, split_name + '_CAPLENS_' + self.data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load bottom up image features distribution
        with open(os.path.join(data_folder, split_name + '_FEATS_' + self.data_name + '.json'), 'r') as j:
            self.objdet = json.load(j)

        # all caption references per image of the current caption
        with open(os.path.join(data_folder, split_name + '_REFS_' + self.data_name + '.json'), 'r') as j:
            self.refs = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):

        objdet = self.objdet[index]

        # Load bottom up image features
        # DIDEC images are all in "t"

        if objdet[0] == "v":
            img = torch.FloatTensor(self.val_features[objdet[1]])
        else:
            img = torch.FloatTensor(self.train_features[objdet[1]])

        caption = torch.LongTensor(self.captions[index])

        caplen = torch.LongTensor([self.caplens[index]])

        # reference counts
        # train {11, 12, 13, 14, 15, 16}
        # val {12, 13, 14, 15, 16}
        # test {13, 14, 15, 16}

        if self.split is 'train':
            return img, caption, caplen

        elif self.split is 'val':
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score

            ref4cap = self.refs[index]
            #print(len(ref4cap), self.max_ref)


            if len(ref4cap) < self.max_ref:
                # add dummy captions for batching, skip in the train and eval code 0s

                ref4cap = ref4cap + (self.max_ref - len(ref4cap)) * [self.max_cap_len * [0]]

            all_captions = torch.LongTensor(ref4cap)

            return img, caption, caplen, all_captions

        elif self.split is 'test':
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score

            ref4cap = self.refs[index]
            # print(len(ref4cap), self.max_ref)

            # here batch size is 1,
            # so no need to add dummy captions for batching, skip in the train and eval code -1s

            ''' if len(ref4cap) < self.max_ref:
                
                ref4cap = ref4cap + (self.max_ref - len(ref4cap)) * [self.max_cap_len * [-1]]'''

            all_captions = torch.LongTensor(ref4cap)

            return img, caption, caplen, all_captions
