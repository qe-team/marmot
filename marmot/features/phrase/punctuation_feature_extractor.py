from __future__ import division

from marmot.features.feature_extractor import FeatureExtractor


class PunctuationFeatureExtractor(FeatureExtractor):

    def __init__(self):
        self.punctuation = ['.', ',', ':', ';', '?', '!']

    def get_features(self, context_obj):
        punct_source, punct_target = [], []
        for punct in self.punctuation:
            tmp_source, tmp_target = 0, 0
            for word in context_obj['token']:
                if word == punct:
                    tmp_target += 1
            for word in context_obj['source_token']:
                if word == punct:
                    tmp_source += 1
            punct_source.append(tmp_source)
            punct_target.append(tmp_target)
        punct_source_norm = [p/len(context_obj['token']) for p in punct_source]
        punct_target_norm = [p/len(context_obj['token']) for p in punct_target]
        return punct_source + punct_source_norm + punct_target + punct_target_norm

    def get_feature_names(self):
        return ['num_period_source',
                'num_commas_source',
                'num_colons_source',
                'num_semicolons_source',
                'num_question_source',
                'num_exclamation_source',
                'num_period_source_weighted',
                'num_commas_source_weighted',
                'num_colons_source_weighted',
                'num_semicolons_source_weighted',
                'num_question_source_weighted',
                'num_exclamation_source_weighted',
                'num_period_target',
                'num_commas_target',
                'num_colons_target',
                'num_semicolons_target',
                'num_question_target',
                'num_exclamation_target',
                'num_period_target_weighted',
                'num_commas_target_weighted',
                'num_colons_target_weighted',
                'num_semicolons_target_weighted',
                'num_question_target_weighted',
                'num_exclamation_target_weighted']
