import string
from marmot.features.feature_extractor import FeatureExtractor


class PunctuationBigramFeatureExtractor(FeatureExtractor):
    '''
    Number of punctuation marks in source and target:
    <source_number>_<target_number>
    '''

    def __init__(self):
        self.punctuation = string.punctuation

    def get_feature(self, context_obj):
        source_punct, target_punct = 0, 0
        for w in context_obj['target']:
            if w in self.punctuation:
                target_punct += 1
        for w in context_obj['source']:
            if w in self.punctuation:
                source_punct += 1

        feature_val = str(source_punct) + "_" + str(target_punct)
        return feature_val

    def get_feature_name(self):
        return "source_target_punctuation_numbers"

    def get_features(self, context_obj):
        return [self.get_feature(context_obj)]

    def get_feature_names(self):
        return [self.get_feature_name()]
