from nltk import word_tokenize

from marmot.features.feature_extractor import FeatureExtractor
from marmot.exceptions.no_data_error import NoDataError


class PseudoReferenceFeatureExtractor(FeatureExtractor):
    '''
    A feature that extracts the pseudo-reference feature 
    for pseudo-references provided in a file 
    (as an alternative to GoogleTranslateFeatureExtractor)
    '''

    def __init__(self, ref_file):
        self.pseudo_references = []
        for line in open(ref_file):
            self.pseudo_references.append(word_tokenize(line[:-1].decode('utf-8')))

    def get_features(self, context_obj):
        if 'sentence_id' not in context_obj:
            raise NoDataError('sentence_id', context_obj, 'PseudoReferenceFeatureExtractor')

        out = 1 if context_obj['token'] in self.pseudo_references[context_obj['sentence_id']] else 0
        return [out]

    def get_feature_names(self):
        return ["pseudo-reference"]
