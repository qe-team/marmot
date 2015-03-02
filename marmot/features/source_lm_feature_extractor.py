import codecs
from subprocess import call
import os
from collections import defaultdict

from marmot.features.feature_extractor import FeatureExtractor
from marmot.util.ngram_window_extractor import extract_window, left_context, right_context
from marmot.experiment.import_utils import mk_tmp_dir


# Class that extracts various LM features for source
class SourceLMFeatureExtractor(FeatureExtractor):

    def __init__(self, ngram_file=None, corpus_file=None, srilm=None, tmp_dir=None, order=3):
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
                self.lm[chunks[0]] == chunks[1]
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
        if ngram in self.lm:
            return 1.0
        elif ngram[:2] in self.lm and ngram[1:] in self.lm:
            return 0.8
        elif ngram[1:] in self.lm:
            return 0.6
        elif ngram[:2] in self.lm and ngram[2] in self.lm:
            return 0.4
        elif ngram[1] in self.lm and ngram[2] in self.lm:
            return 0.3
        elif ngram[2] in self.lm:
            return 0.2
        else:
            return 0.1


    # returns a set of features related to LM
    # currently extracting: highest order ngram including the word and its LEFT context,
    #                       highest order ngram including the word and its RIGHT context
    def get_features(self, context_obj):
        if 'source' not in context_obj:
            raise NoDataError('source', context_obj, 'SourceLMFeatureExtractor')
        if 'alignments' not in context_obj:
            raise NoDataError('alignments', context_obj, 'SourceLMFeatureExtractor')
        align = sorted(context_obj['alignments'][context_obj['index']])
        # unaligned
        if align == []:
            return [0, 0]
        idx_first = align[0]
        idx_last = align[-1]
        words_number = idx_last - idx_first
        tokens = context_obj['source'][idx_first:idx_last+1]

        left_ngram = left_context(context_obj['source'], tokens[0], context_size=self.order-1-words_number, idx=idx_first) + tokens
        right_ngram = tokens + right_context(context_obj['source'], tokens[-1], context_size=self.order-1-words_number, idx=idx_last)
        left_ngram_order = self.check_lm(left_ngram, side='left')
        right_ngram_order = self.check_lm(right_ngram, side='right')


#        left_trigram = left_context(context_obj['target'], context_obj['token'], context_size=2, idx=idx) + [context_obj['token']]
#        middle_trigram = extract_window(context_obj['target'], context_obj['token'], idx=idx)
#        right_trigram = [context_obj['token']] + right_context(context_obj['target'], context_obj['token'], context_size=2, idx=idx)
#
#        backoff_left = self.get_backoff(left_trigram)
 #       backoff_middle = self.get_backoff(middle_trigram)
#        backoff_right = self.get_backoff(right_trigram)

#        return [left_ngram_order, right_ngram_order, backoff_left, backoff_middle, backoff_right]
        return [left_ngram_order, right_ngram_order]



    def get_feature_names(self):
        return ['source_highest_order_ngram_left', 'source_highest_order_ngram_right']
