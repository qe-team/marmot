import codecs
from subprocess import call
import os
from collections import defaultdict

from marmot.features.feature_extractor import FeatureExtractor
from marmot.util.ngram_window_extractor import left_context, right_context
from marmot.experiment.import_utils import mk_tmp_dir
from marmot.exceptions.no_data_error import NoDataError


# Class that extracts various LM features for source
class SourceLMFeatureExtractor(FeatureExtractor):

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

    # returns a set of features related to LM
    # currently extracting: highest order ngram including the word and its LEFT context,
    #                       highest order ngram including the word and its RIGHT context
    def get_features(self, context_obj):
        if 'source' not in context_obj:
            raise NoDataError('source', context_obj, 'SourceLMFeatureExtractor')
        if 'alignments' not in context_obj:
            raise NoDataError('alignments', context_obj, 'SourceLMFeatureExtractor')
        align_idx = context_obj['alignments'][context_obj['index']]
        # unaligned
        if align_idx is None:
            return [0, 0]
        align_token = context_obj['source'][align_idx]

        left_ngram = left_context(context_obj['source'], align_token, context_size=2, idx=align_idx) + [align_token]
        right_ngram = [align_token] + right_context(context_obj['source'], align_token, context_size=2, idx=align_idx)
        left_ngram_order = self.check_lm(left_ngram, side='left')
        right_ngram_order = self.check_lm(right_ngram, side='right')

        return [left_ngram_order, right_ngram_order]

    def get_feature_names(self):
        return ['source_highest_order_ngram_left', 'source_highest_order_ngram_right']
