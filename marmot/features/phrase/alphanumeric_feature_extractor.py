from __future__ import division
import sys
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
        #sys.stderr.write("Start AlphaNumericFeatureExtractor\n")
        tg_numbers = 0
        tg_alphanumeric = 0
        for word in context_obj['token']:
            try:
                float(word)
                tg_numbers += 1
            except:
                if word.isalnum() and not word.isalpha():
                    tg_alphanumeric += 1
        src_numbers = 0
        src_alphanumeric = 0
        if 'source_token' in context_obj and len(context_obj['source_token']) > 0:
            for word in context_obj['source_token']:
                try:
                    float(word)
                    src_numbers += 1
                except:
                    if word.isalnum() and not word.isalpha():
                        src_alphanumeric += 1
 
        src_len = len(context_obj['source_token'])
        tg_len = len(context_obj['token'])
        src_tg_num_diff = abs(src_numbers - tg_numbers)/tg_len
        src_tg_alnum_diff = abs(src_alphanumeric - tg_alphanumeric)/tg_len
        src_num_percent = 0
        src_alnum_percent = 0
        if src_len > 0:
            src_num_percent = src_numbers/src_len
            src_alnum_percent = src_alphanumeric/src_len
 
        all_out = [str(src_num_percent),
                str(tg_numbers/tg_len),
                str(src_tg_num_diff),
                str(src_alnum_percent),
                str(tg_alphanumeric/tg_len),
                str(src_tg_alnum_diff)]
        #sys.stderr.write("Finish AlphaNumericFeatureExtractor\n")
        return all_out

    def get_feature_names(self):
        return ['percentage_src_numbers',
                'percentage_tg_numbers',
                'src_tg_numbers_normalized_diff',
                'percentage_src_alphanumeric',
                'percentage_tg_alphanumeric',
                'src_tg_alphanumeric_normalized_diff']
