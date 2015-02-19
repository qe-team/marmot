from goslate import Goslate
from nltk import word_tokenize

from marmot.features.feature_extractor import FeatureExtractor


class GoogleTranslateFeatureExtractor(FeatureExtractor):

    def __init__(self, lang='en'):
        self.gs = Goslate()
        self.lang = lang

    def get_features(self, context_obj):
        if 'source' not in context_obj:
            print("No source for translation")
            return []
        translation = word_tokenize(self.gs.translate(' '.join(context_obj['source']), self.lang))
        if context_obj['token'] in translation:
            return [1]
        return [0]

    def get_feature_names(self):
        return ["pseudo_reference"]
