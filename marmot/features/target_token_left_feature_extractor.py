from marmot.features.feature_extractor import FeatureExtractor
from marmot.util.ngram_window_extractor import left_context, right_context


class TargetTokenLeftFeatureExtractor(FeatureExtractor):
    '''
    Target features:
      - target token
      - left and right windows of the target token
    '''

    def __init__(self, context_size=1):
        self.context_size = context_size

    def get_features(self, context_obj):
        token = context_obj['token']
        left = ' '.join(left_context(context_obj['target'], token, context_size=self.context_size, idx=context_obj['index']))
        return [token, left]

    def get_feature_names(self):
        return ['token', 'left_context']
