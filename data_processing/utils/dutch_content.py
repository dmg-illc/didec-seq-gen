import json
import spacy
from collections import defaultdict

nlp = spacy.load('nl_core_news_sm')

with open('../data/split_train.json', 'r') as file:
    split_train = json.load(file)

with open('../data/split_val.json', 'r') as file:
    split_val = json.load(file)

with open('../data/split_test.json', 'r') as file:
    split_test = json.load(file)


def get_content_seqs(split_set, nlp):

    # dictionaries to keep content sets for img-ppn pairs
    noun_set_split = defaultdict(list) # NOUNS ONLY
    nav_set_split = defaultdict(list) # NOUNS ADJECTIVES VERBS

    count = 0

    # sort the dict keys in case python might return them in a different order
    # participants are kept in a list, which is always sorted

    for im in sorted(split_set.keys()):

        for p in split_set[im]:

            noun_set = [] # specific to img-ppn pair
            nav_set = [] # specific to img-ppn pair

            count+=1

            if count % 100 == 0:
                print(count)

            # GETTING THE CAPTION IN THE GRAMMARS, THEY ARE THE MOST PROCESSED ONES
            grammar_file = '../data/grammars/raw_caption_' + p + '_' + im + '.jsgf'

            with open(grammar_file, 'r') as file:
                grammar_lines = file.readlines()

            for line in grammar_lines:
                if '=' in line:
                    split_line = line.split('=')

                    caption = split_line[1].split(';')[0]

                    # checked lowercase...
                    # checked inital space...

                    # if not caption.islower(): # islower true if none is upper
                    #     print('UPPERCASE', caption)
                    #
                    # if not caption[0] == ' ':
                    #     print('NONSPACE', caption) # not tokenizing the caption, but there is a space at the beginning from the grammar

                    caption = caption[1:] #skip the space at the beginning

                    doc = nlp(caption)

                    #print(doc.text)

                    for token in doc:
                        #only the nouns
                        if token.pos_ == 'NOUN':
                            #print(token.text, token.pos_)
                            noun_set.append(token.text)
                            nav_set.append(token.text)

                        elif token.pos_ == 'ADJ' or token.pos_ == 'VERB':
                            nav_set.append(token.text)

                    #print()

            noun_set_split[im].append(noun_set)
            nav_set_split[im].append(nav_set)

    return noun_set_split, nav_set_split


print('train')
noun_set_train, nav_set_train = get_content_seqs(split_train, nlp)

print('val')
noun_set_val, nav_set_val = get_content_seqs(split_val, nlp)

print('test')
noun_set_test, nav_set_test = get_content_seqs(split_test, nlp)

print('saving noun sets')
with open('../data/nounset_train_didec.json', 'w') as f:
    json.dump(noun_set_train, f)

with open('../data/nounset_val_didec.json', 'w') as f:
    json.dump(noun_set_val, f)

with open('../data/nounset_test_didec.json', 'w') as f:
    json.dump(noun_set_test, f)

print('saving nav sets')
with open('../data/navset_train_didec.json', 'w') as f:
    json.dump(nav_set_train, f)

with open('../data/navset_val_didec.json', 'w') as f:
    json.dump(nav_set_val, f)

with open('../data/navset_test_didec.json', 'w') as f:
    json.dump(nav_set_test, f)