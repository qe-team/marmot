from marmot.features.feature_extractor import FeatureExtractor
from marmot.util.ngram_window_extractor import left_context


class TrilexicalLeftFeatureExtractor(FeatureExtractor):
    '''
    Trilexical features:
        - target token + left context + source token
        - target token + right context + source token
    '''

    def get_features(self, context_obj):
        token = context_obj['token']
        left = ' '.join(left_context(context_obj['target'], token, context_size=1, idx=context_obj['index']))

        align_idx = context_obj['alignments'][context_obj['index']]
        if align_idx is None:
            aligned_to = '__unaligned__'
        else:
            aligned_to = context_obj['source'][align_idx]

        return [token + '|' + left + '|' + aligned_to]

    def get_feature_names(self):
        return ['target+left+source']
