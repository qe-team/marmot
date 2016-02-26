from __future__ import print_function

import sys
from nltk import wordnet as wn
from collections import defaultdict

from marmot.features.feature_extractor import FeatureExtractor


class WordnetFeatureExtractor(FeatureExtractor):

    def __init__(self, src_lang, tg_lang, wordnet_pos=None):
        # there's a bug in nltk: with lang = 'eng' some words are not found, but with lang = 'en' they are,
        # and in the wordnet.langs() 'en' is not listed, only 'eng'
        if src_lang != 'en' and src_lang not in wn.wordnet.langs():
            print('The language', src_lang, 'is not supported by Open Multilingual Wordnet\n')
            self.src_lang = ''
        else:
            self.src_lang = src_lang
        if tg_lang != 'en' and tg_lang not in wn.wordnet.langs():
            print('The language', tg_lang, 'is not supported by Open Multilingual Wordnet\n')
            self.tg_lang = ''
        else:
            self.tg_lang = tg_lang

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
        # TODO: should it throw an error when src or alignments don't exist?
        if self.src_lang == '' or 'alignments' not in context_obj or 'source' not in context_obj or context_obj['alignments'][context_obj['index']] == None:
            src_count = 0
        else:
            src_align = context_obj['alignments'][context_obj['index']]
            src_token = context_obj['source'][src_align]
            src_count = len(wn.wordnet.synsets(src_token, lang=self.src_lang))
        if self.tg_lang == '':
            tg_count = 0
        else:
            tg_count = len(wn.wordnet.synsets(context_obj['token'], lang=self.tg_lang))
        return [src_count, tg_count]

    def get_feature_names(self):
        return ["polysemy_count_source", "polysemy_count_target"]
