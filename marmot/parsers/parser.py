from abc import ABCMeta, abstractmethod
# A parser takes one or more filenames and (optionally) keys
# returns an object containing keys which each point to a list of lists

class Parser(object):

    __metaclass__ = ABCMeta

    # subclasses must provide the implementation
    # the flexible args and kwargs are not ideal here, but we need to keep parsers very flexible
    # another problem with this approach is that we don't need to initialize most parsers
    # it might be better to use a @classmethod, or @staticmethod
    @abstractmethod
    def parse(self, *args, **kwargs):
        pass
