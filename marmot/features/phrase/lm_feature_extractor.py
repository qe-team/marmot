from __future__ import division

import kenlm
from marmot.features.feature_extractor import FeatureExtractor


class LMFeatureExtractor(FeatureExtractor):

    def __init__(self, lm_file):
        self.model = kenlm.LanguageModel(lm_file)

    def get_features(self, context_obj):
        log_prob = self.model.score(' '.join(context_obj['token']), bos=False, eos=False)
        tg_len = len(context_obj['token'])
        perplexity = 2**((-1/tg_len)*log_prob)
        return [log_prob, perplexity]

    def get_feature_names(self):
        return ['target_log_prob', 'target_perplexity']
