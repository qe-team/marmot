from __future__ import print_function, division
import numpy as np
from marmot.features.feature_extractor import FeatureExtractor
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('testlogger')

class TokenCountFeatureExtractor(FeatureExtractor):

    # extract the word2vec features for a window of tokens around the target token
    def get_features(self, context_obj):
        feature_funcs = [self.source_count, self.target_count, self.source_target_ratio]
        return np.array([f(context_obj) for f in feature_funcs])

    def source_count(self, context_obj):
        if 'source' in context_obj and type(context_obj['source']) == list:
            return float(len(context_obj['source']))
        logger.warn('you are trying to extract the source token count from a context object without a "source" field')
        return 0.0


    def target_count(self, context_obj):
        if 'target' in context_obj and type(context_obj['target'] == list):
            return float(len(context_obj['target']))
        logger.warn('you are trying to extract the target token count from a context object without a "target" field')
        return 0.0

    def source_target_ratio(self, context_obj):
        s_count = self.source_count(context_obj)
        t_count = self.target_count(context_obj)
        return s_count / t_count



