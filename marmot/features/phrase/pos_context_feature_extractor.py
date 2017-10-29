from marmot.features.feature_extractor import FeatureExtractor
from marmot.exceptions.no_data_error import NoDataError
from marmot.util.ngram_window_extractor import left_context, right_context


class POSContextFeatureExtractor(FeatureExtractor):

    def get_features(self, context_obj):
        if 'target_pos' not in context_obj:
            raise NoDataError('target_pos', context_obj, 'POSContextFeatureExtractor')
        if 'source_pos' not in context_obj:
            raise NoDataError('source_pos', context_obj, 'POSContextFeatureExtractor')

        left_src = left_context(context_obj['source_pos'], context_obj['source_pos'][context_obj['source_index'][0]], context_size=1, idx=context_obj['source_index'][0])
        right_src = right_context(context_obj['source_pos'], context_obj['source_pos'][context_obj['source_index'][1]-1], context_size=1, idx=context_obj['source_index'][1]-1)

        left_tg = left_context(context_obj['target_pos'], context_obj['target_pos'][context_obj['index'][0]], context_size=1, idx=context_obj['index'][0])
        right_tg = right_context(context_obj['target_pos'], context_obj['target_pos'][context_obj['index'][1]-1], context_size=1, idx=context_obj['index'][1]-1)

        return [left_src[0], right_src[0], left_tg[0], right_tg[0]]

    def get_feature_names(self):
        return ['left_source_context_pos', 'right_source_context_pos', 'left_target_context_pos', 'right_target_context_pos']
