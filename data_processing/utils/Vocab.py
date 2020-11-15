import os
import pickle
import csv

from collections import Counter
from utils.UtteranceTokenizer import UtteranceTokenizer


class Vocab():

    def __init__(self, file, min_occ=3):
        print("Initialising vocab from file.")

        self.word2index = {}
        self.index2word = {}
        self.word2count = {}

        for t in ['<pad>', '<unk>', '<A>', '<B>', '-A-', '-B-']:
            self.index2word[len(self.word2index)] = t
            self.word2index[t] = len(self.word2index)

        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                w, c = row[0], int(row[1])
                if c >= min_occ:
                    self.word2index[w] = len(self.word2index)
                    self.index2word[self.word2index[w]] = w
                    self.word2count[w] = c

    def __len__(self):
        return len(self.word2index)

    def __getitem__(self, q):
        if isinstance(q, str):
            return self.word2index.get(q, self.word2index['<unk>'])
        elif isinstance(q, int):
            return self.index2word.get(q, self.index2word[self.word2index['<unk>']])
        else:
            raise ValueError("Expected str or int but got {}".format(type(q)))

    def encode(self, x):
        return [self[xi] for xi in x]

    def decode(self, x):
        return [self[xi] for xi in x]

    @classmethod
    def create(cls, data_path, data_file, vocab_file, tokenization, lowercase, splitting, min_occ=3):
        """
        Creates a vocabulary from the given PhotoBook data set
        :param data_path:
        :param data_file:
        :param vocab_file:
        :param min_occ:
        :return:
        """
        print("Creating new vocab from {}".format(data_file))

        with open(os.path.join(data_path, data_file), 'rb') as f:
            games = pickle.load(f)

        tokenizer = UtteranceTokenizer()

        # Gather word token frequencies
        tokens = []
        for _, game_segments in games:
            for round_segments, _ in game_segments:
                for (segment, _) in round_segments:
                    for message in segment:
                        if message.type == "text":
                            tokens.extend([t for t in tokenizer.tokenize_utterance(message.text, tokenization, lowercase, splitting)])

        # Determine occurrence cutoff
        token_counter = Counter(tokens).most_common()
        cutoff = None
        for idx, element in enumerate(token_counter):
            if element[1] < min_occ:
                cutoff = idx
                break

        print("Done.")

        word_list = []
        for idx, (word, count) in enumerate(token_counter):
            if idx == cutoff: break
            word_list.append((word, count))
        with open(os.path.join(data_path, vocab_file), "w") as f:
            writer = csv.writer(f, delimiter=',', quotechar='|')
            writer.writerows(word_list)

        return cls(os.path.join(data_path, vocab_file), min_occ)
