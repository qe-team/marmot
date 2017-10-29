from __future__ import division

import sys
from marmot.features.feature_extractor import FeatureExtractor


class PunctuationFeatureExtractor(FeatureExtractor):

    def __init__(self):
        self.punctuation = ['.', ',', ':', ';', '?', '!']

    def get_features(self, context_obj):
        #sys.stderr.write("Start PunctuationFeatureExtractor\n")
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
        target_len = len(context_obj['token'])
        punct_diff = [(src - tg) for (src, tg) in zip(punct_source, punct_target)]
        punct_diff_norm = [(src - tg)/target_len for (src, tg) in zip(punct_source, punct_target)]
        other_features = []
        if len(context_obj['source_token']) > 0:
            other_features.append(sum(punct_source)/len(context_obj['source_token']))
        else:
            other_features.append(0)
        other_features.append(sum(punct_target)/len(context_obj['token']))
        other_features.append((sum(punct_source) - sum(punct_target))/target_len)
        #sys.stderr.write("Finish PunctuationFeatureExtractor\n")
        return [str(p) for p in punct_diff] + [str(p) for p in punct_diff_norm] + [str(p) for p in other_features]

    def get_feature_names(self):
        return ['diff_periods',
                'diff_commas',
                'diff_colons',
                'diff_semicolons',
                'diff_questions',
                'diff_exclamations',
                'diff_periods_weighted',
                'diff_commas_weighted',
                'diff_colons_weighted',
                'diff_semicolons_weighted',
                'diff_questions_weighted',
                'diff_exclamations_weighted',
                'percentage_punct_source',
                'percentage_punct_target',
                'diff_punct']
