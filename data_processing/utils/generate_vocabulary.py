import json
from collections import Counter
from collections import defaultdict
from utils.UtteranceTokenizer import UtteranceTokenizer

#VOCAB FROM TRAIN SPLIT
# only DIDEC words

with open('../data/split_train.json', 'r') as file:
    split_train = json.load(file)

all_text = []

tokenizer = UtteranceTokenizer('nl')
lowercase = True
method = 'nltk'
min_occurrence = 1

'''
in nltk 's included such as auto's not auto 's 
<corr>	693
<pause>	123
<rep>	139
<uh>	1277
'''

count = 0

for im in split_train:

    for p in split_train[im]:
        count+=1

        # GETTING THE CAPTION IN THE GRAMMARS, THEY ARE THE MOST PROCESSED ONES
        grammar_file = '../data/grammars/raw_caption_' + p + '_' + im + '.jsgf'

        with open(grammar_file, 'r') as file:
            grammar_lines = file.readlines()

        for line in grammar_lines:
            if '=' in line:
                split_line = line.split('=')

                caption = split_line[1].split(';')[0]

                tokenized_caption = tokenizer.tokenize_utterance(utterance=caption, method=method,
                                                                   lowercase=lowercase)

                all_text.extend(tokenized_caption)

print(count)

token_counter = Counter(all_text).most_common()
print(token_counter)

tokens2write = defaultdict(int)


#add <unk> <pad> <start> <end>

tokens2write['<pad>'] = 0
tokens2write['<unk>'] = 1
tokens2write['<start>'] = 2
tokens2write['<end>'] = 3

vocab_count = 4

for t in token_counter:
    token, freq = t

    if freq < min_occurrence:
        pass #these are unks

    else:
        tokens2write[token] = vocab_count
        vocab_count += 1

print(vocab_count)


with open('../data/WORDMAP_didec.json', 'w') as file:
    json.dump(tokens2write, file)


