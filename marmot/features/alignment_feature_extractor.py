from __future__ import print_function

import os
import errno

from marmot.features.feature_extractor import FeatureExtractor
from marmot.util.alignments import train_alignments, align_sentence
from marmot.util.ngram_window_extractor import left_context, right_context


# all features that require source dictionary
class AlignmentFeatureExtractor(FeatureExtractor):

    def __init__(self, align_model='', src_file='', tg_file='', tmp_dir=None, context_size=1):
        if tmp_dir is None:
            tmp_dir = os.getcwd()
        try:
            os.makedirs(tmp_dir)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(tmp_dir):
                pass
            else:
                raise
        self.tmp_dir = tmp_dir

        self.model = ''

        # no alignment model
        if align_model == '':
            # if src_file and tg_file are not empty, it means that an alignment model needs to be trained
            # (self.model doesn't have to be defined, if context objects have alignments)
            if os.path.isfile(src_file) and os.path.isfile(tg_file):
                self.model = train_alignments(src_file, tg_file, self.tmp_dir)
        else:
            self.model = align_model
        self.context_size = context_size

    def get_features(self, context_obj):
        if 'source' not in context_obj or 'target' not in context_obj or context_obj['source'] is None or context_obj['target'] is None:
            print('Source and/or target sentences are not defined. Not enough information for alignment features extraction')
            return []

        if 'alignments' not in context_obj:
            if self.model == '':
                print('No alignment model specified and no pre-defined alignments in context object')
                return []
            context_obj['alignments'] = align_sentence(context_obj['source'], context_obj['target'], self.model)

        # source word(s)
        source_nums = context_obj['alignments'][context_obj['index']]
        # if word is unaligned - no source and no source contexts
        if source_nums == []:
            return ['Unaligned', ['Unaligned' for i in range(self.context_size)], ['Unaligned' for i in range(self.context_size)]]

        # TODO: find contexts for all words aligned to the token (now only 1st word)
        else:
            left = left_context(context_obj['source'], context_obj['source'][source_nums[0]], context_size=self.context_size, idx=source_nums[0])
            right = right_context(context_obj['source'], context_obj['source'][source_nums[0]], context_size=self.context_size, idx=source_nums[0])

        return [context_obj['source'][source_nums[0]], left, right]

    def get_feature_names(self):
        return ['first_aligned_token', 'left_alignment', 'right_alignment']
