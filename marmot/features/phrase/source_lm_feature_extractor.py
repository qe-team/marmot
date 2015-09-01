import kenlm
from marmot.features.feature_extractor import FeatureExtractor


class SourceLMFeatureExtractor(FeatureExtractor):

    def __init__(self, lm_file):
        self.model = kenlm.LanguageModel(lm_file)

    def get_features(self, context_obj):
        if 'source_token' not in context_obj or len(context_obj['source_token'] == 0):
            return 0
        return self.model.score(' '.join(context_obj['source_token']))

    def get_feature_names(self):
        return ['source_log_prob']
