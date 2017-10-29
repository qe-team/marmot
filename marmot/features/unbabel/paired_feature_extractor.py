from marmot.features.feature_extractor import FeatureExtractor
from marmot.util.ngram_window_extractor import left_context, right_context


class PairedFeatureExtractor(FeatureExtractor):
    '''
    Paired features:
        - target token + left context
        - target token + right context
        - target token + source token
        - target POS + source POS
    '''

    def get_features(self, context_obj):
        token = context_obj['token']
        left = ' '.join(left_context(context_obj['target'], token, context_size=1, idx=context_obj['index']))
        right = ' '.join(right_context(context_obj['target'], token, context_size=1, idx=context_obj['index']))
        tg_pos = context_obj['target_pos'][context_obj['index']] if context_obj['target_pos'] != [] else ''

        align_idx = context_obj['alignments'][context_obj['index']]
        if align_idx is None:
            src_token = '__unaligned__'
            src_pos = '__unaligned__'
        else:
            src_token = context_obj['source'][align_idx]
            src_pos = context_obj['source_pos'][align_idx]

        return [token + '|' + left, token + '|' + right, token + '|' + src_token, tg_pos + '|' + src_pos]

    def get_feature_names(self):
        return ['token+left', 'token+right', 'token+source', 'POS+sourcePOS']
