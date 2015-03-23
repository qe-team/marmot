import sys
from subprocess import Popen, PIPE

from marmot.features.feature_extractor import FeatureExtractor
from marmot.exceptions.no_data_error import NoDataError
from marmot.exceptions.no_resource_error import NoResourceError


class POSFeatureExtractor(FeatureExtractor):
    """
    POS for source and target words, tagged with TreeTagger
    """
    def __init__(self, tagger=None, par_file_src=None, par_file_tg=None):
        self.tagger = tagger
        self.par_src = par_file_src
        self.par_tg = par_file_tg

    # tag words if context_obj has no tagging
    # returns tags for all words in sentence
    def _call_tagger(self, tok_list, lang='tg'):
        par_file = self.par_tg if lang == 'tg' else self.par_src
        out = []

        if self.tagger is None:
            raise NoResourceError('tagger', 'POSFeatureExtractor')
        if par_file is None:
            raise NoResourceError('tagging parameters', 'POSFeatureExtractor')

        p = Popen([self.tagger, '-quiet', par_file], stdin=PIPE, stdout=PIPE)
        out = p.communicate(input='\n'.join([tok.encode('utf-8') for tok in tok_list]))[0].decode('utf-8').split('\n')
        return out

    def get_features(self, context_obj):
        if 'target_pos' not in context_obj:
            if 'target' in context_obj and context_obj['target'] is not None:
                context_obj['target_pos'] = self._call_tagger(context_obj['target'])
            else:
                raise NoDataError('target_pos', context_obj, 'POSFeatureExtractor')
        if 'source_pos' not in context_obj:
            if 'source' in context_obj and context_obj['source'] is not None:
                context_obj['source_pos'] = self._call_tagger(context_obj['source'], lang='src')
            else:
                raise NoDataError('source_pos', context_obj, 'POSFeatureExtractor')

        # extract POS features:
        # - target POS
        # - source POS (may be more than 1)
        # - something else?
        tg_pos = context_obj['target_pos'][context_obj['index']] if context_obj['target_pos'] != [] else ''
        src_pos = []
        if 'source_pos' in context_obj and context_obj['source_pos'] != [] and 'alignments' in context_obj:
            src_pos = [context_obj['source_pos'][i] for i in context_obj['alignments'][context_obj['index']]]
            src_pos = ' '.join(src_pos)

        return [tg_pos, src_pos]

    def get_feature_names(self):
        return ['target_pos', 'aligned_source_pos_list']
