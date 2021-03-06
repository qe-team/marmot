import os
import sys
import codecs
from subprocess import call
from collections import defaultdict
from marmot.features.feature_extractor import FeatureExtractor
from marmot.util.ngram_window_extractor import left_context, right_context
from marmot.experiment.import_utils import mk_tmp_dir


class ContextLMLeftFeatureExtractor(FeatureExtractor):
    '''
    Same as ContextLMFeatureExtractor, but without right context
    '''

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

    def get_features(self, context_obj):
        #sys.stderr.write("Start ContextLMLeftFeatureExtractor\n")
        idx_left = context_obj['index'][0]
        idx_right = context_obj['index'][1]

        left_ngram = left_context(context_obj['target'], context_obj['token'][0], context_size=self.order-1, idx=idx_left) + [context_obj['token'][0]]
        left_ngram_order = self.check_lm(left_ngram, side='left')

        left_trigram = left_context(context_obj['target'], context_obj['token'][0], context_size=2, idx=idx_left) + [context_obj['token'][0]]

        backoff_left = self.get_backoff(left_trigram)

        #sys.stderr.write("Finish ContextLMLeftFeatureExtractor\n")
        return [str(left_ngram_order), str(backoff_left)]

    def get_feature_names(self):
        return ['highest_order_ngram_left', 'backoff_behavior_left']
