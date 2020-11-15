from nltk import TweetTokenizer
import spacy
from spacy.lang.nl import Dutch
from spacy.lang.en import English

class UtteranceTokenizer():
    def __init__(self, language='nl'):
        self.tweet_tokenizer = TweetTokenizer()

        if language == 'nl':
            # dutch
            self.spacy_nlp = spacy.load('nl')
            self.spacy_tokenizer = Dutch().Defaults.create_tokenizer(self.spacy_nlp)

        elif language == 'en':
            # english
            self.spacy_nlp = spacy.load('en')
            self.spacy_tokenizer = English().Defaults.create_tokenizer(self.spacy_nlp)


    def tokenize_utterance(self, utterance, method = 'nltk', lowercase=True):
        """

        Tokenises a given utterance
        :param utterance: String. Utterance to be tokenised
        :param lowercase: bool. Set True to lowercase the utterance before processing it
        :return: list. A list of word tokens from the utterance
        """
        if lowercase:
            utterance = utterance.lower()

        tokens = []

        if method == 'nltk':
            tokens = self.tweet_tokenizer.tokenize(utterance)

        elif method == 'spacy':

            # CHECK SPACY VS NLTK

            tokens = self.spacy_tokenizer(utterance)

            for t in tokens:
                print(t)
        else:
            print('Method not defined')

        return tokens
