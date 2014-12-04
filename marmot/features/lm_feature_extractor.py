from nltk import ngrams, word_tokenize
#from nltk.model import NgramModel

from marmot.features.feature_extractor import FeatureExtractor
from marmot.util.simple_corpus import SimpleCorpus


# Class that extracts various LM features 
# Calling an external LM is very slow, so a new lm is constructed with nltk 
class LMFeatureExtractor(FeatureExtractor):

    def __init__(self, corpus_file, order=3):

        self.order = order
        self.lm = [ set() for i in range(order) ]
        for line in open(corpus_file):
             words = word_tokenize(line[:-1].decode('utf-8'))
             for i in range(1,order):
                 self.lm[i] = self.lm[i].union( ngrams( words, i+1 ) )
             self.lm[0] = self.lm[0].union(words)
    
    def check_lm(self, ngram, side='left'):
        for i in range(self.order, 0, -1):
            if side == 'left':
                cur_ngram = ngram[len(ngram)-i:]
            elif side == 'right':
                cur_ngram = ngram[:i]
            if tuple(cur_ngram) in self.lm[i-1]:
                return i
        return 0


  # returns a set of features related to LM
  # currently extracting: highest order ngram including the word and its LEFT context,
  #                       highest order ngram including the word and its RIGHT context
    def get_features(self, context_obj):
        left_ngram = self.check_lm( context_obj['target'][:context_obj['index']+1], side='left' )
        right_ngram = self.check_lm( context_obj['target'][context_obj['index']:], side='right' )
        return (left_ngram, right_ngram)
