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
    # a context obj looks like:
    # { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
    # some fields MAY BE MISSING from a given context object, the implementation needs to check for its fields
    @abstractmethod
    def get_features(self, context_obj):
        pass
