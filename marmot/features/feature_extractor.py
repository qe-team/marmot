# an abstract class representing a feature extractor
# a feature extractor takes an object like
# { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
# and extracts features for that instance

# IMPORTANT - different feature extractors require different fields in the context object
# - it's up to the feature extractor implementation to determine which fields it actually needs, and to ensure that the object contains them

from abc import ABCMeta, abstractmethod

# this is an abstract class which extracts contexts according to a user-provided implementation
# a negative context is a context that is representative of a WRONG usage of a word
# a negative context for a word may have nothing to do with a positive context (i.e. it may just be random)


class FeatureExtractor(object):

    __metaclass__ = ABCMeta

    # subclasses must provide the implementation
    @abstractmethod
    def get_features(self, context_obj):
        """
        returns a list of features (one or more)
        :param context_obj: { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>, ...}
        :return: [<feature1>, <feature2>, ...]
        - some fields MAY BE MISSING from a given context object, the implementation needs to check for its fields
        """
        pass

    @abstractmethod
    def get_feature_names(self):
        """
        :return: a list of strings representing names of the features returned by get_features
        """
        pass
