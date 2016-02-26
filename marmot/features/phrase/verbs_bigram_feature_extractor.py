from collections import defaultdict
from marmot.features.feature_extractor import FeatureExtractor


def get_verbs(language):
    verbs = defaultdict(str)
    verbs['english'] = ['VB']
    verbs['spanish'] = ['VL']
    return verbs[language]


class VerbsBigramFeatureExtractor(FeatureExtractor):
    '''
    Number of punctuation marks in source and target:
    <source_number>_<target_number>
    '''

    def __init__(self, lang='english'):
        self.verbs = get_verbs(lang)

    def is_verb(self, word):
        for n in self.verbs:
            if word.startswith(n):
                return True
        return False

    def get_feature(self, context_obj):
        source_idx = context_obj['source_index']
        target_idx = context_obj['index']
        source_verbs, target_verbs = 0, 0
        for w in context_obj['target_pos'][target_idx[0]:target_idx[1]]:
            if self.is_verb(w):
                target_verbs += 1
        for w in context_obj['source_pos'][source_idx[0]:source_idx[1]]:
            if self.is_verb(w):
                source_verbs += 1

        feature_val = str(source_verbs) + "_" + str(target_verbs)
        return feature_val

    def get_feature_name(self):
        return "source_target_verbs_numbers"

    def get_features(self, context_obj):
        return [self.get_feature(context_obj)]

    def get_feature_names(self):
        return [self.get_feature_name()]
