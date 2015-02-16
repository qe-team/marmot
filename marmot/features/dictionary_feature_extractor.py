import string
import nltk

from marmot.features.feature_extractor import FeatureExtractor


class DictionaryFeatureExtractor(FeatureExtractor):
    '''
    Extract binary features indicating that the word belongs to a list of special tokens:
    - stopwords
    - punctuation symbols
    - proper names
    - numbers
    '''

    def __init__(self, language='', stopwords=[], punctuation=[], proper=[]):
        # all lists can be defined by user, otherwise are taken from python and nltk
        self.punctuation = punctuation if punctuation else string.punctuation
        self.stopwords = stopwords if stopwords else nltk.corpus.stopwords.words(language) if language else nltk.corpus.stopwords.words()
        self.proper = proper

    # returns:
    #    ( is stopword, is punctuation, is proper name, is digit )
    def get_features(self, context_obj):
        tok = context_obj['token']
        return [int(tok in self.stopwords), int(tok in self.punctuation), int(tok in self.proper if self.proper else tok.istitle()), int(tok.isdigit())]

    def get_feature_names(self):
        return ['is_stopword', 'is_punctuation', 'is_proper_noun', 'is_digit']
