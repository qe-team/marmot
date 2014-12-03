from nltk.model import NgramModel

from marmot.features.feature_extractor import FeatureExtractor
from marmot.util.simple_corpus import SimpleCorpus

def check_lm_recursive(words, lm, low_order='left'):
  if len(words) < lm._n:
    return check_lm_recursive(words, lm._backoff, low_order=low_order)    

  if tuple(words) in lm._ngrams:
    return lm._n
  elif lm._n > 1:
    if low_order == 'left':  
      return check_lm_recursive(words[1:], lm._backoff, low_order=low_order)
    elif low_order == 'right':
      return check_lm_recursive(words[:-1], lm._backoff, low_order=low_order)
  else:
    return 0


# Class that extracts various LM features 
# Calling an external LM is very slow, so a new lm is constructed with nltk 
class LMFeatureExtractor(FeatureExtractor):

  def __init__(self, corpus_file, order=3):
    # load the corpus
    corp = SimpleCorpus(corpus_file)
    # nltk LM requires all words in one list
    all_words = [w for sent in [line for line in corp.get_texts()] for w in sent]
    self.lm = NgramModel(order, all_words)


  def check_lm_recursive(words, lm, low_order='left'):
    if len(words) < lm._n:
      return check_lm_recursive(words, lm._backoff, low_order=low_order)    

    if tuple(words) in lm._ngrams:
      return lm._n
    elif lm._n > 1:
      if low_order == 'left':  
        return check_lm_recursive(words[1:], lm._backoff, low_order=low_order)
      elif low_order == 'right':
        return check_lm_recursive(words[:-1], lm._backoff, low_order=low_order)
    else:
      return 0


  # returns a set of features related to LM
  # currently extracting: highest order ngram including the word and its LEFT context,
  #                       highest order ngram including the word and its RIGHT context
  def get_features(self, context_obj):
    left_ngram = check_lm_recursive(context_obj['target'][max(0, context_obj['index']-self.lm._n):context_obj['index']], self.lm, low_order='left')
    right_ngram = check_lm_recursive(context_obj['target'][context_obj['index']:min(len(context_obj['target']),context_obj['index']+self.lm._n)], self.lm, low_order='right')
    return (left_ngram, right_ngram)
