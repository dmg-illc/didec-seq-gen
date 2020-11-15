# Adapted from https://github.com/poojahira/image-captioning-bottom-up-top-down/blob/master/models.py
# to incorporate the DIDEC input
# Top-down attention module is fed features of an image masked with sequential fixation masks
# aligned with the utterances of words

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision
from torch.nn.utils.weight_norm import weight_norm

class Attention(nn.Module):
    def __init__(self, features_dim, decoder_dim, attention_dim, dropout=0.5):

        """
        :param features_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()

        self.features_att = weight_norm(nn.Linear(features_dim, attention_dim))  # linear layer to transform encoded image
        self.decoder_att = weight_norm(nn.Linear(decoder_dim, attention_dim))  # linear layer to transform decoder's output
        self.full_att = weight_norm(nn.Linear(attention_dim, 1))  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, image_features, decoder_hidden):

        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, 36, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """

        att1 = self.features_att(image_features)  # (batch_size, 36, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(2)  # (batch_size, 36)
        alpha = self.softmax(att)  # (batch_size, 36)
        attention_weighted_encoding = (image_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, features_dim)

        return attention_weighted_encoding


class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, attention_dim, features_dim, dropout, device):
        super(DecoderWithAttention, self).__init__()

        self.device = device

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.features_dim = features_dim

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0, scale_grad_by_freq=True)

        self.language_model = nn.LSTMCell(self.features_dim + self.hidden_dim, self.hidden_dim, bias=True)  # language model LSTMCell

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.attention = Attention(self.features_dim, self.hidden_dim, self.attention_dim)

        self.top_down_attention = nn.LSTMCell(self.embedding_dim + self.features_dim + self.hidden_dim, self.hidden_dim, bias=True) # top down attention LSTMCell
        self.fc = weight_norm(nn.Linear(self.hidden_dim, vocab_size))  # linear layer to find scores over vocabulary

        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, batch_size):

        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros(batch_size, self.hidden_dim).to(self.device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        return h, c

    def forward(self, image_feats, segment, lengths, word_fxs):
        """
        @param image_feats: bottom-up image features
        @param segment: caption text converted into indices
        @param lengths: contains the caption lengths
        @param word_fxs: features of images masked with sequential fixation masks aligned with words
        """
        batch_size = segment.shape[0]

        # pack sequence
        sorted_lengths, sorted_idx = torch.sort(lengths.squeeze(1), descending=True)

        image_feats = image_feats[sorted_idx]
        segment = segment[sorted_idx]
        word_fxs = word_fxs[sorted_idx]

        embeds_words = self.embedding(segment)  # b, l, d

        embeds_words = self.dropout(embeds_words)

        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (sorted_lengths - 1).tolist()
        maxd = max(decode_lengths)

        # Create tensors to hold word prediction scores
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(self.device)

        # At each time-step, pass the language model's previous hidden state, the mean pooled bottom up features and
        # word embeddings to the top down attention model. Then pass the hidden state of the top down model and the bottom up
        # features to the attention block. The attention weighed bottom up features and hidden state of the top down attention model
        # are then passed to the language model

        for t in range(max(decode_lengths)):

            batch_size_t = sum([l > t for l in decode_lengths])

            # top-down attention LSTM directly receives the features of an image masked
            # with sequential fixations that are aligned with words being fed at the same timestep

            h1, c1 = self.top_down_attention(
                torch.cat([h2[:batch_size_t], word_fxs[:batch_size_t, t, :], embeds_words[:batch_size_t, t, :]], dim=1), (h1[:batch_size_t], c1[:batch_size_t]))

            attention_weighted_encoding = self.attention(image_feats[:batch_size_t], h1[:batch_size_t])

            h2, c2 = self.language_model(
                torch.cat([attention_weighted_encoding[:batch_size_t], h1[:batch_size_t]], dim=1),
                (h2[:batch_size_t], c2[:batch_size_t]))

            preds = self.fc(self.dropout(h2))  # (batch_size_t, vocab_size)

            predictions[:batch_size_t, t, :] = preds

        return predictions, segment, decode_lengths, sorted_idx
