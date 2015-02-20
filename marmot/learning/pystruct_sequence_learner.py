import numpy as np

from marmot.learning.sequence_learner import SequenceLearner
from pystruct.models import ChainCRF
from pystruct.learners import OneSlackSSVM
from pystruct.learners import StructuredPerceptron

# a learner which uses the pystruct library
class PystructSequenceLearner(SequenceLearner):
    def __init__(self):
        # the model
        self.model = ChainCRF(directed=True)
        # the learner
        self.learner = OneSlackSSVM(model=self.model, C=.1, inference_cache=50, tol=0.1, n_jobs=1)
        # self.learner = StructuredPerceptron(model=self.model, average=True)

    def fit(self, X, y):
        # Train linear chain CRF
        self.learner.fit(X, y)

    def predict(self, X):
        return self.learner.predict(X)



