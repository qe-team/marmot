from __future__ import division

import numpy as np
from collections import defaultdict
from nltk import FreqDist
from gensim.corpora import TextCorpus

from marmot.features.feature_extractor import FeatureExtractor


class NumTranslationsFeatureExtractor(FeatureExtractor):

    # .f2e file
    def __init__(self, lex_prob_file, corpus_file):
        self.lex_prob = defaultdict(list)
        for line in open(lex_prob_file):
            chunks = line[:-1].split()
            self.lex_prob[chunks[1]].append(float(chunks[2]))
        corpus = TextCorpus(input=corpus_file)
        self.corpus_freq = FreqDist([word for line in corpus.get_texts() for word in line])
        self.thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]

    def get_features(self, context_obj):
        if 'source_token' not in context_obj or len(context_obj['source_token']) == 0:
            return [0.0 for i in range(len(self.thresholds)*2)]

        translations, translations_weighted = [], []
        for thr in self.thresholds:
            all_words, all_words_weighted = [], []
            for word in context_obj['source_token']:
                trans = [fl for fl in self.lex_prob[word] if fl >= thr]
                all_words.append(len(trans))
                all_words_weighted.append(len(trans)*self.corpus_freq.freq(word))
            translations.append(np.average(all_words))
            translations_weighted.append(np.average(all_words_weighted))
        return translations + translations_weighted

    def get_feature_names(self):
        return ['source_translations_001_freq',
                'source_translations_005_freq',
                'source_translations_01_freq',
                'source_translations_02_freq',
                'source_translations_05_freq',
                'source_translations_001_freq_weighted',
                'source_translations_005_freq_weighted',
                'source_translations_01_freq_weighted',
                'source_translations_02_freq_weighted',
                'source_translations_05_freq_weighted']
