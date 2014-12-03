from __future__ import print_function, division
import numpy as np
from word_level.features.feature_extractor import FeatureExtractor


class TokenCountFeatureExtractor(FeatureExtractor):

    # extract the word2vec features for a window of tokens around the target token
    def get_features(self, context_obj):
        feature_funcs = [self.source_count, self.target_count, self.source_target_ratio]
        return np.array([f(context_obj) for f in feature_funcs])

    def source_count(self, context_obj):
        try:
            return float(len(context_obj['source']))
        except ValueError as e:
            print(e)
            raise

    def target_count(self, context_obj):
        try:
            return float(len(context_obj['target']))
        except ValueError as e:
            print(e)
            raise

    def source_target_ratio(self, context_obj):
        s_count = self.source_count(context_obj)
        t_count = self.target_count(context_obj)
        return s_count / t_count



