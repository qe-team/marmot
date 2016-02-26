import numpy as np
import sys

import logging
from gensim.models import Word2Vec
from marmot.features.feature_extractor import FeatureExtractor

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('experiment_logger')


def left_context(token_list, token, context_size, idx):
    left_window = []
    if idx <= 0:
        #return ['_START_' for i in range(context_size)]
        return ['<s>' for i in range(context_size)]
    assert(token_list[idx] == token)
    for i in range(idx-context_size, idx):
        if i < 0:
            #left_window.append('_START_')
            left_window.append('<s>')
        else:
            left_window.append(token_list[i])
    return left_window


def right_context(token_list, token, context_size, idx):
    right_window = []
    if idx >= len(token_list):
        #return ['_END_' for i in range(context_size)]
        return ['</s>' for i in range(context_size)]
    assert(token_list[idx] == token), "Token in token list: {}, index: {}, token provided in parameters: {}".format(token_list[idx], idx, token)
    for i in range(idx+1, idx+context_size+1):
        if i > len(token_list)-1:
            #right_window.append('_END_')
            right_window.append('</s>')
        else:
            right_window.append(token_list[i])
    return right_window


class Word2VecFeatureExtractor(FeatureExtractor):
    '''
    Combine a feature vector for an ngram of arbitrary length
    from w2v vectors of all words of the ngram.
    <combination> --- method of combination of word vectors:
        - 'sum' (default)
        - 'avg'
    '''

    def __init__(self, w2v_file, combination='sum', context_size=2):

        self.model = Word2Vec.load(w2v_file)
        self.default_vector = np.average(np.array([self.model[x] for x in self.model.vocab]), axis=0).reshape((-1,))
        self.zero_vector = np.zeros(self.default_vector.shape[0])
        self.context_size = context_size
        if combination == 'sum':
            self.combine = np.sum
        elif combination == 'avg':
            self.combine = np.average
        else:
            print("Unknown combination type provided: '{}'".format(combination))

    def extract_word2vec_vector(self, token):
        if token in self.model.vocab:
            return self.model[token]
        #elif token == '_START_' or token == '_END_':
        elif token == '<s>' or token == '</s>':
            return self.zero_vector
        else:
            return self.default_vector

    # extract the word2vec features for a window of tokens around the target token
    def get_features(self, context_obj):
        if 'token' not in context_obj or len(context_obj['token']) == 0:
            print("No token in context object ", context_obj)
            sys.exit()
        if 'index' not in context_obj or len(context_obj['index']) == 0:
            print("No index in context object ", context_obj)
            sys.exit()

        if context_obj['index'][0] == context_obj['index'][1]:
            print("Invalid token indices in sentence: ", context_obj['target'])
            print("Indices: {}, {}".format(context_obj['index'][0], context_obj['index'][1]))

        phrase_vector = []
        # if 'token' contains more than 1 string, 'index' should be an interval
        if type(context_obj['token']) is list or type(context_obj['token']) is np.ndarray:
            left_window = left_context(context_obj['target'], context_obj['token'][0], self.context_size, context_obj['index'][0])
            right_window = right_context(context_obj['target'], context_obj['token'][-1], self.context_size, context_obj['index'][-1]-1)
            phrase_vector = [self.extract_word2vec_vector(tok) for tok in context_obj['token']]
            phrase_vector = self.combine(phrase_vector, axis=0)
        else:
            left_window = left_context(context_obj['target'], context_obj['token'], self.context_size, context_obj['index'])
            right_window = right_context(context_obj['target'], context_obj['token'], self.context_size, context_obj['index'])
            phrase_vector = self.extract_word2vec_vector(context_obj['token'])

        vector = []
        for tok in left_window:
            vector.extend(self.extract_word2vec_vector(tok))
        vector.extend(phrase_vector)
        for tok in right_window:
            vector.extend(self.extract_word2vec_vector(tok))
        return np.hstack(vector)

    # TODO: there should be a name for every feature
    def get_feature_names(self):
        return ['w2v'+str(i) for i in range(len(self.default_vector)*(self.context_size*2 + 1))]
