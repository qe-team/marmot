from __future__ import print_function
import codecs
from subprocess import call
import os
from collections import defaultdict

from marmot.features.feature_extractor import FeatureExtractor
from marmot.util.ngram_window_extractor import extract_window, left_context, right_context
from marmot.experiment.import_utils import mk_tmp_dir


# Class that extracts various LM features
# Calling an external LM is very slow, so a new lm is constructed with nltk
class LMFeatureExtractor(FeatureExtractor):

    def __init__(self, ngram_file=None, corpus_file=None, srilm=None, tmp_dir=None, order=5):
        # generate ngram counts
        if ngram_file is None:
            if srilm is None:
                if 'SRILM' in os.environ:
                    srilm = os.environ['SRILM']
                else:
                    print("No SRILM found")
                    return
            if corpus_file is None:
                print ("No corpus for LM generation")
                return

            srilm_ngram_count = os.path.join(srilm, 'ngram-count')

            tmp_dir = mk_tmp_dir(tmp_dir)
            lm_file = os.path.join(tmp_dir, 'lm_file')
            ngram_file = os.path.join(tmp_dir, 'ngram_count_file')
            call([srilm_ngram_count, '-text', corpus_file, '-lm', lm_file, '-order', str(order), '-write', ngram_file])

        self.lm = defaultdict(int)
        for line in codecs.open(ngram_file, encoding='utf-8'):
            chunks = line[:-1].split('\t')
            if len(chunks) == 2:
                new_tuple = tuple(chunks[0].split())
                new_number = int(chunks[1])
                self.lm[new_tuple] = new_number
            else:
                print("Wrong ngram-counts file format at line '", line[:-1], "'")

        self.order = order

    def check_lm(self, ngram, side='left'):
        for i in range(self.order, 0, -1):
            if side == 'left':
                cur_ngram = ngram[len(ngram)-i:]
            elif side == 'right':
                cur_ngram = ngram[:i]
            else:
                print("Unknown parameter 'side'", side)
                return 0
            if tuple(cur_ngram) in self.lm:
                return i
        return 0

    def get_backoff(self, ngram):
        assert(len(ngram) == 3)
        ngram = tuple(ngram)
        # trigram (1, 2, 3)
        if ngram in self.lm:
            return 1.0
        # two bigrams (1, 2) and (2, 3)
        elif ngram[:2] in self.lm and ngram[1:] in self.lm:
            return 0.8
        # bigram (2, 3)
        elif ngram[1:] in self.lm:
            return 0.6
        # bigram (1, 2) and unigram (3)
        elif ngram[:2] in self.lm and ngram[2:] in self.lm:
            return 0.4
        # unigrams (2) and (3)
        elif ngram[1:2] in self.lm and ngram[2:] in self.lm:
            return 0.3
        # unigram (3)
        elif ngram[2:] in self.lm:
            return 0.2
        # all words unknown
        else:
            return 0.1


    # returns a set of features related to LM
    # currently extracting: highest order ngram including the word and its LEFT context,
    #                       highest order ngram including the word and its RIGHT context
    def get_features(self, context_obj):
        idx = context_obj['index']

        left_ngram = left_context(context_obj['target'], context_obj['token'], context_size=self.order-1, idx=idx) + [context_obj['token']]
        right_ngram = [context_obj['token']] + right_context(context_obj['target'], context_obj['token'], context_size=self.order-1, idx=idx)
        left_ngram_order = self.check_lm(left_ngram, side='left')
        right_ngram_order = self.check_lm(right_ngram, side='right')

        left_trigram = left_context(context_obj['target'], context_obj['token'], context_size=2, idx=idx) + [context_obj['token']]
        middle_trigram = extract_window(context_obj['target'], context_obj['token'], idx=idx)
        right_trigram = [context_obj['token']] + right_context(context_obj['target'], context_obj['token'], context_size=2, idx=idx)
        # TODO: instead of _START_ there should be <s>

        backoff_left = self.get_backoff(left_trigram)
        backoff_middle = self.get_backoff(middle_trigram)
        backoff_right = self.get_backoff(right_trigram)

        return [left_ngram_order, right_ngram_order, backoff_left, backoff_middle, backoff_right]

    def get_feature_names(self):
        return ['highest_order_ngram_left', 'highest_order_ngram_right', 'backoff_behavior_left', 'backoff_behavior_middle', 'backoff_behavior_right']
