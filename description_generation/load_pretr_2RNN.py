import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch import nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.utils.data

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nlgeval import NLGEval

import time

import os
import json

import argparse

import datetime

from models.model_pret_gaze_2RNN import DecoderWithAttention

from utils.DidecDataset import DidecDataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_folder", type=str, default='final_dataset')
    parser.add_argument("-data_name", type=str, default='didec')
    parser.add_argument("-emb_dim", type=int, default=1024)
    parser.add_argument("-attention_dim", type=int, default=1024)
    parser.add_argument("-decoder_dim", type=int, default=1024)
    parser.add_argument("-features_dim", type=int, default=2048)
    parser.add_argument("-dropout", type=float, default=0.0)
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-break_flag", action='store_true')
    parser.add_argument("-shuffle_flag", action='store_true')
    parser.add_argument("-num_workers", type=int, default=0)
    parser.add_argument("-print_flag", action='store_true')
    parser.add_argument("-print_freq", type=int, default=10)
    parser.add_argument("-checkpoint", type=str, default=None)
    parser.add_argument("-learning_rate", type=float, default=0.02)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-top_x", type=int, default=5)
    parser.add_argument("-metric", type=str, default='cider')

    args = parser.parse_args()

    print(args)

    model_type = '2RNN_DV'

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    data_folder = args.data_folder
    data_name = args.data_name

    batch_size = args.batch_size
    break_flag = args.break_flag  # to break after 5 batches, for debugging reasons
    shuffle_flag = args.shuffle_flag  # both train and val
    print_flag = args.print_flag  # prints target and hypotheses

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)

    cudnn.benchmark = True

    emb_dim = args.emb_dim  # dimension of word embeddings
    attention_dim = args.attention_dim  # dimension of attention linear layers
    decoder_dim = args.decoder_dim  # dimension of decoder RNN
    features_dim = args.features_dim #img dim
    learning_rate = args.learning_rate  # learning rate for the optimizer
    dropout = args.dropout

    top_x = args.top_x

    start_epoch = 0
    epochs = args.epochs  # number of epochs to train for (if early stopping is not triggered)
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU

    best_bleu4 = 0.0  # to store the best BLEU-4 score
    best_cider = 0.0 # to store the best CIDER score
    best_loss = 0.0 # to store the best Cross-Entropy loss

    smoothing_method = SmoothingFunction().method1  # epsilon method for bleu

    print_freq = args.print_freq  # print training/validation stats every __ batches
    checkpoint = args.checkpoint  # path to checkpoint, None if none

    # Read word map
    # WARNING union vocab from pretrained + didec itself
    word_map_file = os.path.join(data_folder, 'WORDMAP_union.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    rev_word_map = {v: k for k, v in word_map.items()}

    print('vocab len', len(word_map))

    i2w = dict()

    for w in word_map:
        i2w[word_map[w]] = w

    print('using a pretrained checkpoint')

    checkpoint = torch.load(checkpoint, map_location=device)

    # these were not in pre-trained

    gaze_rnn = nn.LSTMCell(features_dim, decoder_dim, bias=True)
    hid2feat = nn.Linear(decoder_dim, features_dim)

    hid2feat.bias.data.fill_(0)
    hid2feat.weight.data.uniform_(-0.1, 0.1)

    checkpoint['model_state_dict'].update({'gaze_rnn.bias_ih': gaze_rnn.bias_ih})

    checkpoint['model_state_dict'].update({'gaze_rnn.bias_hh': gaze_rnn.bias_hh})

    checkpoint['model_state_dict'].update({'gaze_rnn.weight_ih': gaze_rnn.weight_ih})

    checkpoint['model_state_dict'].update({'gaze_rnn.weight_hh': gaze_rnn.weight_hh})

    checkpoint['model_state_dict'].update({'hid2feat.bias': hid2feat.bias})

    checkpoint['model_state_dict'].update({'hid2feat.weight': hid2feat.weight})

    # construct the model and optimizer with values from args

    decoder = DecoderWithAttention(vocab_size=len(word_map), embedding_dim=emb_dim, hidden_dim=decoder_dim,
                                   attention_dim=attention_dim, features_dim=features_dim, dropout=dropout, device=device)

    # fill in state dicts
    decoder.load_state_dict(checkpoint['model_state_dict'])

    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    decoder = decoder.to(device)