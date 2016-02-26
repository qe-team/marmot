from __future__ import print_function
import sys
from marmot.features.feature_extractor import FeatureExtractor


class PrevWordFeatureExtractor(FeatureExtractor):
    '''
    Extract next word
    '''
    def get_feature(self, context_obj):

        if type(context_obj['index']) is int:
            last_word_idx = context_obj['index']
        elif type(context_obj['index']) is tuple:
            last_word_idx = context_obj['index'][1]
        else:
            print("Unknown type of context object's 'index' field: {}".format(type(context_obj['index'])))
            sys.exit()

        next_word = context_obj['target'][last_word_idx] if last_word_idx < len(context_obj['target']) else "</s>"
        return next_word

    def get_feature_name(self):
        return "next_word"

    def get_features(self, context_obj):
        return [self.get_feature(context_obj)]

    def get_feature_names(self):
        return [self.get_feature_name()]
