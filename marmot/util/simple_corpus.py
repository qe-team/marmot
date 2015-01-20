#!/usr/bin/env python
#encoding: utf-8

from __future__ import division, print_function
from gensim import utils, corpora, models, similarities
import numpy as np
import codecs

from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from scipy import sparse

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('testlogger')

class SimpleCorpus(corpora.TextCorpus):

    # make sure that the token2id dict gets created
    def __init__(self, corpus_file):
        # logger.info('corpus file:'
        # print(type(corpus_file))
        corpus_file = codecs.open(corpus_file, encoding='utf8')
        super(SimpleCorpus, self).__init__(corpus_file)
        self.dictionary.id2token = {v: k for k,v in self.dictionary.token2id.items()}


    def get_texts(self):
        """
        Parse documents from the file provided in the constructor.
        format: one document per line.
        nltk is used for tokenization
        """
        with self.getstream() as stream:
            for doc in stream:
                # yield [word for word in word_tokenize(utils.to_unicode(doc).lower())]
                yield [word for word in word_tokenize(utils.to_unicode(doc))]

    
    def get_texts_raw(self):
        """
        Parse documents analogously to SimpleCorpus.get_texts(),
        but tokenized by whitespace only
        """
        wst = WhitespaceTokenizer()
        with self.getstream() as stream:
            for doc in stream:
                # yield [word for word in word_tokenize(utils.to_unicode(doc).lower())]
                yield [word for word in wst.tokenize(utils.to_unicode(doc))]


    def __len__(self):
        """Define this so we can use `len(corpus)`"""
        if 'length' not in self.__dict__:
            self.length = sum(1 for doc in self.get_texts())
        return self.length

    # get a one-hot representation as a numpy array using the corpus indices
    def binary_vec_from_list(self, token_list):
        vec = np.zeros(len(self.dictionary.keys()), dtype=np.int)
        if token_list is not None:
            for tok in token_list:
                try:
                    col_idx = self.dictionary.token2id[tok]
                    vec[col_idx] = 1
                except KeyError:
                    pass
        vvec = sparse.csr_matrix( vec )
#        print( "Binary vector: ", vvec.shape )
        return vvec

# build a corpus from a file (one document per line)
def build_corpus(filename):
    return SimpleCorpus(filename)



