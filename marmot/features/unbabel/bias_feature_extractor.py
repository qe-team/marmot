from marmot.features.feature_extractor import FeatureExtractor


class BiasFeatureExtractor(FeatureExtractor):

    def get_features(self, context_obj):
        return [1]

    def get_feature_names(self):
        return ['bias']
