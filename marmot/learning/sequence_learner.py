# this is an abstract class representing a sequence learner, or 'structured' learner
# implementations wrap various sequence learning tools, in order to provide a consistent interface within Marmot

from abc import ABCMeta, abstractmethod

class SequenceLearner(object):

    __metaclass__ = ABCMeta

    # subclasses must provide the implementation
    @abstractmethod
    def fit(self, X, y):
        '''
        fit a sequence model to data in the format [[seq1_w1, seq1_w2, ...]],
        :param X: a list of np.arrays, where each row in each array contains the features for an item in the sequence - X can be viewed as a 3d tensor
        :param y: the true labels for each sequence
        :return:
        '''
        pass

    @abstractmethod
    def predict(self, X):
        '''
        predict the tag for each item in each sequence
        :param X: list of sequences list of np.array
        :return: list of lists, where each list contains the predictions for the test sequence
        '''
        pass