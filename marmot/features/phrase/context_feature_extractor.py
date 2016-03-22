from marmot.features.feature_extractor import FeatureExtractor
from marmot.util.ngram_window_extractor import left_context, right_context


class ContextFeatureExtractor(FeatureExtractor):

    def get_features(self, context_obj):
        if 'source_token' in context_obj:
            left_src = left_context(context_obj['source'], context_obj['source_token'][0], context_size=1, idx=context_obj['source_index'][0])
            right_src = right_context(context_obj['source'], context_obj['source_token'][-1], context_size=1, idx=context_obj['source_index'][1]-1)
        else:
            left_src = ""
            right_src = ""
        left_tg = left_context(context_obj['target'], context_obj['token'][0], context_size=1, idx=context_obj['index'][0])
        right_tg = right_context(context_obj['target'], context_obj['token'][-1], context_size=1, idx=context_obj['index'][1]-1)

        return [left_src[0], right_src[0], left_tg[0], right_tg[0]]

    def get_feature_names(self):
        return ['left_source_context', 'right_source_context', 'left_target_context', 'right_target_context']
