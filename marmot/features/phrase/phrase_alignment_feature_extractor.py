from __future__ import division, print_function

import sys
import numpy as np
import os
import errno

from marmot.features.feature_extractor import FeatureExtractor
from marmot.util.alignments import train_alignments, align_sentence
from marmot.exceptions.no_data_error import NoDataError


class PhraseAlignmentFeatureExtractor(FeatureExtractor):
    '''
    Extract phrase-level alignment features:
     - percentage of unaligned words
     - percentage of words with more than 1 aligned words
     - average number of aligned words per word
     - ...?
    '''

    def __init__(self, align_model='', src_file='', tg_file='', tmp_dir=None):
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

    def get_features(self, context_obj):
        #sys.stderr.write("Start PhraseAlignmentFeatureExtractor\n")
        if 'source' not in context_obj or context_obj['source'] is None:
            #sys.stderr.write('No source')
            raise NoDataError('source', context_obj, 'AlignmentFeatureExtractor')
        if 'target' not in context_obj or context_obj['source'] is None or context_obj['target'] is None:
            #sys.stderr.write('No target')
            raise NoDataError('target', context_obj, 'AlignmentFeatureExtractor')

        if 'alignments_all' not in context_obj:
            context_obj['alignments_all'] = [[i] for i in context_obj['alignments']]
            #raise NoDataError('alignments_all', context_obj, 'AlignmentFeatureExtractor')
#        if self.model == '':
#            raise NoDataError('alignments', context_obj, 'AlignmentFeatureExtractor')
        # we have to extract new alignments because we need the number of aligned words per target word
#        local_alignments = align_sentence(context_obj['source'], context_obj['target'], self.model)
        n_unaligned, n_multiple = 0, 0
        n_alignments = []
        #sys.stderr.write('All fine\n')
        #sys.stderr.write('%s\n' % (', '.join([s for s in context_obj])))
        #sys.stderr.write('%s, %i\n' % (type(context_obj['index']), len(context_obj['index'])))
        #sys.stderr.write('Context obj index: %i to %i\n' % (context_obj['index'][0], context_obj['index'][1]))
        for i in range(context_obj['index'][0], context_obj['index'][1]):
            assert(all([w == ww for (w, ww) in zip(context_obj['token'], [context_obj['target'][j] for j in range(context_obj['index'][0], context_obj['index'][1])])])), "Assertion failed"
            #sys.stderr.write('Assertion was fine\n')
            #print(context_obj['alignments_all'])
            cur_alignments = len(context_obj['alignments_all'][i])
            #sys.stderr.write('Alignments_all\n')
            if cur_alignments == 0:
                #sys.stderr.write('Cur_alignments = 0\n')
                n_unaligned += 1
            elif cur_alignments > 1:
                #sys.stderr.write('Cur_alignments > 1\n')
                n_multiple += 1
            #sys.stderr.write('Op!\n')
            n_alignments.append(cur_alignments)

        #sys.stderr.write('Still fine')
        tg_len = len(context_obj['token'])
        #sys.stderr.write("Finish PhraseAlignmentFeatureExtractor\n")
        return [str(n_unaligned/tg_len), str(n_multiple/tg_len), str(np.average(n_alignments))]

    def get_feature_names(self):
        return ['num_unaligned', 'num_multi_alignment', 'avg_alignments_num']
