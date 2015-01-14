import numpy as np
import sys
from marmot.features.feature_extractor import FeatureExtractor

from scipy.stats import norm

#generate features from distributions of some existing set of examples
class GaussianFeatureExtractor(FeatureExtractor):

    def __init__(self, features):
        # the distribution for each feature - pair (mean, std.deviation)
        # TODO: there is an error here - we want the distribution for each feature, this overwrites every time
        self.distributions = {tok: (np.average(vec), np.std(vec)) for tok, feature_vectors in features.items() for vec in feature_vectors.T}

    #context_obj may contain only token
    def get_features(self, context_obj):
        token = context_obj['token']
        if not self.distributions.has_key(token):
            sys.stderr.write('No distribution for token %s\n' % token.encode('utf-8'))
            return []
        return np.array([norm.rvs(loc=avg, scale=std) for avg,std in self.distributions[token]])

