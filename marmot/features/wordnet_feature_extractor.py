import sys
from nltk import wordnet as wn
from collections import defaultdict

from marmot.features.feature_extractor import FeatureExtractor


class WordnetFeatureExtractor(FeatureExtractor):

    def __init__(self, lang='en', wordnet_pos=None):
        # there's a bug in nltk: with lang = 'eng' some words are not found, but with lang = 'en' they are,
        # and in the wordnet.langs() 'en' is not listed, only 'eng'
        if lang != 'en' and lang not in wn.wordnet.langs():
            sys.stderr.write('The language is not supported by Open Multilingual Wordnet\n')
            self.lang = ''
        else:
            self.lang = lang
        # mapping between parts of speech returned by a POS-tagger and WordNet parts of speech:
        #   wn.ADJ, wn.ADV, wn.NOUN, wn.VERB etc.
        if wordnet_pos is not None:
            self.pos_dict = defaultdict(lambda: None)
            for line in open(wordnet_pos):
                a_pair = line[:-1].decode('utf-8').split('\t')
                if len(a_pair) != 2:
                    sys.stderr.write('Incorrect format of the mapping file')
                self.pos_dict[a_pair[0]] = a_pair[1]
        else:
            self.pos_dict = None

    def get_features(self, context_obj):
        if self.pos_dict is not None and 'target_pos' in context_obj:
            cur_pos = context_obj['target_pos'][context_obj['index']]
            return len(wn.wordnet.synsets(context_obj['token'], pos=self.pos_dict[cur_pos], lang=self.lang))
        else:
            return len(wn.wordnet.synsets(context_obj['token'], lang=self.lang))

    def get_feature_names(self):
        return ["polysemy count"]
