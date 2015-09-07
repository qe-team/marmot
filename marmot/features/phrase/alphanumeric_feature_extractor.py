from __future__ import division
from marmot.features.feature_extractor import FeatureExtractor


class AlphaNumericFeatureExtractor(FeatureExtractor):
    '''
    - percentage of numbers in the source
    - percentage of numbers in the target
    - absolute difference between number of numbers in the source and target sentence normalised by source sentence length
    - percentage of source words that contain non-alphabetic symbols
    - percentage of target words that contain non-alphabetic symbols
    - ratio of percentage of tokens a-z in the source and tokens a-z in the target
    '''

    def get_features(self, context_obj):
        tg_numbers = 0
        tg_alphanumeric = 0
        for word in context_obj['token']:
            if word.isalnum():
                tg_alphanumeric += 1
            try:
                float(word)
                tg_numbers += 1
            except:
                pass
        src_numbers = 0
        src_alphanumeric = 0
        for word in context_obj['source_token']:
            if word.isalnum():
                tg_alphanumeric += 1
            try:
                float(word)
                tg_numbers += 1
            except:
                pass

        src_len = len(context_obj['source_token'])
        tg_len = len(context_obj['token'])
        return [src_numbers/src_len,
                tg_numbers/tg_len,
                (src_numbers - tg_numbers)/src_len,
                src_alphanumeric/src_len,
                tg_alphanumeric/tg_len,
                (src_alphanumeric - tg_alphanumeric)/src_len]

    def get_feature_names(self):
        return ['percentage_src_numbers',
                'percentage_tg_numbers',
                'src_tg_numbers_normalized_ratio',
                'percentage_src_alphanumeric',
                'percentage_tg_alphanumeric',
                'src_tg_alphanumeric_normalized_ratio']
