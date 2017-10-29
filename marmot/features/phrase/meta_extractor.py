import sys


class MetaExtractor():
    '''
    class which applies all feature extractors to an object
    '''

    def __init__(self, extractors):
        sys.stderr.write('This is MetaExtractor init\n')
        self.extractors = extractors

    def get_features(self, context_obj):
        features = []
        for ext in self.extractors:
            features.extend(ext.get_features(context_obj))
        return features

    def get_feature_names(self):
        feature_names = []
        for ext in self.extractors:
            feature_names.extend(ext.get_feature_names())
        return feature_names
