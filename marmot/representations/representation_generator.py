# an abstract class representing a representation generator
# returns the data object
# { representation_name: representation}
# <representation_name> -- string
# <representation> -- list of lists, representation of the whole dataset

from abc import ABCMeta, abstractmethod


class RepresentationGenerator(object):

    __metaclass__ = ABCMeta

    # subclasses must provide the implementation
    # generators may need a "persist = True/False"
    @abstractmethod
    def generate(self, *args, **kwargs):
        pass
