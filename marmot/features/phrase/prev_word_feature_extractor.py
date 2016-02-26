from __future__ import print_function
import sys
from marmot.features.feature_extractor import FeatureExtractor


class PrevWordFeatureExtractor(FeatureExtractor):
    '''
    Extract previous word
    '''
    def get_feature(self, context_obj):

        if type(context_obj['index']) is int:
            first_word_idx = context_obj['index']
        elif type(context_obj['index']) is tuple:
            first_word_idx = context_obj['index'][0]
        else:
            print("Unknown type of context object's 'index' field: {}".format(type(context_obj['index'])))
            sys.exit()

        prev_word = context_obj['target'][first_word_idx - 1] if first_word_idx > 0 else "<s>"
        return prev_word

    def get_feature_name(self):
        return "previoust_word"

    def get_features(self, context_obj):
        return [self.get_feature(context_obj)]

    def get_feature_names(self):
        return [self.get_feature_name()]
