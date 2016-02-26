from collections import defaultdict
from marmot.features.feature_extractor import FeatureExtractor


def get_nouns(language):
    nouns = defaultdict(str)
    nouns['english'] = ['NN']
    nouns['spanish'] = ['NC', 'NMEA', 'NMON', 'NP']
    return nouns[language]


class VerbsBigramFeatureExtractor(FeatureExtractor):
    '''
    Number of punctuation marks in source and target:
    <source_number>_<target_number>
    '''

    def __init__(self, lang='english'):
        self.nouns = get_nouns(lang)

    def is_noun(self, word):
        for n in self.nouns:
            if word.startswith(n):
                return True
        return False

    def get_feature(self, context_obj):
        source_idx = context_obj['source_index']
        target_idx = context_obj['index']
        source_nouns, target_nouns = 0, 0
        for w in context_obj['target_pos'][target_idx[0]:target_idx[1]]:
            if self.is_noun(w):
                target_nouns += 1
        for w in context_obj['source_pos'][source_idx[0]:source_idx[1]]:
            if self.is_noun(w):
                source_nouns += 1

        feature_val = str(source_nouns) + "_" + str(target_nouns)
        return feature_val

    def get_feature_name(self):
        return "source_target_nouns_numbers"

    def get_features(self, context_obj):
        return [self.get_feature(context_obj)]

    def get_feature_names(self):
        return [self.get_feature_name()]
