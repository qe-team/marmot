from abc import ABCMeta, abstractmethod

# this is an abstract class which extracts contexts according to a user-provided implementation
# a negative context is a context that is representative of a WRONG usage of a word
# a negative context for a word may have nothing to do with a positive context (i.e. it may just be random)

class ContextCreator(object):

    __metaclass__ = ABCMeta

    # subclasses must provide the implementation
    @abstractmethod
    def get_contexts(self, token, max_size=None):
        pass

