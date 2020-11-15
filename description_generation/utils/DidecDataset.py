import os
import json
import torch
from collections import defaultdict
from torch.utils.data import Dataset
import h5py

class DidecDataset(Dataset):

    def __init__(self, data_folder, data_name, split, model_type):

        self.max_ref = 16 #max ref per img in all the splits

        self.model_type = model_type

        self.data_folder = data_folder
        self.split = split
        assert self.split in {'train', 'val', 'test'}

        if self.split == 'train':
            self.max_cap_len = 55  # max caption length in the train split
        elif self.split == 'val':
            self.max_cap_len = 44  # max caption length in the val split
        elif self.split == 'test':
            self.max_cap_len = 42  # max caption length in the test split


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


        if model_type == 'SM_FV':

            task_feature_file = os.path.join(data_folder, self.split + '_MASKFEATS_free_view_' + self.data_name + '.json')

            with open(task_feature_file, 'r') as file:

                self.task_img_features = json.load(file)

        elif model_type == 'SM_DV':

            task_feature_file = os.path.join(data_folder, self.split + '_MASKFEATS_description_' + self.data_name + '.json')

            with open(task_feature_file, 'r') as file:

                self.task_img_features = json.load(file)


        elif model_type == 'RNN_DV' or model_type == '2RNN_DV':

            incremental_feature_file = os.path.join(data_folder,
                                             self.split + '_ALGFEATS_' + self.data_name + '.json')

            with open(incremental_feature_file, 'r') as file:

                self.incremental_features = json.load(file)


        elif model_type == 'COMB_FD':

            task_feature_file = os.path.join(data_folder,
                                             self.split + '_MASKFEATS_free_view_' + self.data_name + '.json')

            with open(task_feature_file, 'r') as file:

                self.task_img_features = json.load(file)

            incremental_feature_file = os.path.join(data_folder,
                                                    self.split + '_ALGFEATS_' + self.data_name + '.json')
            with open(incremental_feature_file, 'r') as file:

                self.incremental_features = json.load(file)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):

        objdet = self.objdet[index]

        # Load bottom up image features
        if objdet[0] == "v":
            img = torch.FloatTensor(self.val_features[objdet[1]])
        else:
            img = torch.FloatTensor(self.train_features[objdet[1]])

        caption = torch.LongTensor(self.captions[index])

        caplen = torch.LongTensor([self.caplens[index]])

        if self.model_type == 'SM_DV' or self.model_type == 'SM_FV':

            masked_features = torch.FloatTensor(self.task_img_features[index])

        elif self.model_type == 'RNN_DV' or self.model_type == '2RNN_DV':

            incremental_features = torch.FloatTensor(self.incremental_features[index])

        elif self.model_type == 'COMB_FD':

            masked_features = torch.FloatTensor(self.task_img_features[index])
            incremental_features = torch.FloatTensor(self.incremental_features[index])


        # reference counts
        # train {11, 12, 13, 14, 15, 16}
        # val {12, 13, 14, 15, 16}
        # test {13, 14, 15, 16}

        if self.split == 'train':
            if self.model_type == 'SM_DV' or self.model_type == 'SM_FV':
                return img, caption, caplen, masked_features
            elif self.model_type == 'RNN_DV' or self.model_type == '2RNN_DV':
                return img, caption, caplen, incremental_features
            elif self.model_type == 'COMB_FD':
                return img, caption, caplen, masked_features, incremental_features
            else:
                return img, caption, caplen

        elif self.split == 'val':
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score

            ref4cap = self.refs[index]
            #print(len(ref4cap), self.max_ref)


            if len(ref4cap) < self.max_ref:
                # add dummy captions for batching, skip in the train and eval code 0s

                print("shorter")
                ref4cap = ref4cap + (self.max_ref - len(ref4cap)) * [self.max_cap_len * [0]]
            else:
                print("topmax")

            all_captions = torch.LongTensor(ref4cap)

            if self.model_type == 'SM_DV' or self.model_type == 'SM_FV':
                return img, caption, caplen, all_captions, masked_features
            elif self.model_type == 'RNN_DV' or self.model_type == '2RNN_DV':
                return img, caption, caplen, all_captions, incremental_features
            elif self.model_type == 'COMB_FD':
                return img, caption, caplen, all_captions, masked_features, incremental_features
            else:
                return img, caption, caplen, all_captions

        elif self.split == 'test':
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score

            ref4cap = self.refs[index]
            # print(len(ref4cap), self.max_ref)

            # here batch size is 1,
            # so no need to add dummy captions for batching, skip in the train and eval code -1s

            ''' if len(ref4cap) < self.max_ref:
                
                ref4cap = ref4cap + (self.max_ref - len(ref4cap)) * [self.max_cap_len * [-1]]'''

            all_captions = torch.LongTensor(ref4cap)

            if self.model_type == 'SM_DV' or self.model_type == 'SM_FV':
                return img, caption, caplen, all_captions, masked_features
            elif self.model_type == 'RNN_DV' or self.model_type == '2RNN_DV':
                return img, caption, caplen, all_captions, incremental_features
            elif self.model_type == 'COMB_FD':
                return img, caption, caplen, all_captions, masked_features, incremental_features
            else:
                return img, caption, caplen, all_captions
