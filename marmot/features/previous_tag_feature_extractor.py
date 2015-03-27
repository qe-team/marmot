from __future__ import print_function

from marmot.features.feature_extractor import FeatureExtractor
from marmot.exceptions.no_data_error import NoDataError


class PreviousTagFeatureExtractor(FeatureExtractor):
    '''
    Extracts the tag of the previous token
    '''

    def __init__(self):
        pass

    def get_features(self, context_obj):
        if 'sequence_tags' not in context_obj:
            raise NoDataError('sequence_tags', context_obj, 'PreviousTagFeatureExtractor')

        idx = context_obj['index']
        if idx == 0:
            return ['_START_']
        else:
            return [context_obj['sequence_tags'][idx-1]]

    def get_feature_names(self):
        return ['previous_token_tag']
