import os
from subprocess import Popen, PIPE

from marmot.features.feature_extractor import FeatureExtractor

class POSFeatureExtractor(FeatureExtractor):
  '''
  POS for source and target words, performed with TreeTagger
  '''
  def __init__(self, tagger='', par_file_src='', par_file_tg=''):
    self.tagger = tagger
    self.par_src = par_file_src
    self.par_tg = par_file_tg

  # tag words if context_obj has no tagging
  # returns tags for all words in sentence
  def _call_tagger(self, tok_list, lang='tg'):
    par_file = self.par_tg if lang == 'tg' else self.par_src
    out = []

    if self.tagger == '' or par_file == '':
      sys.stderr.write('Tagging script and parameter file should be provided\n')
      return []

    p = Popen([self.tagger, '-quiet', par_file], stdin=PIPE, stdout=PIPE)
    out = p.communicate(input='\n'.join([tok.encode('utf-8') for tok in tok_list]))[0].decode('utf-8').split('\n')
    return out


  def get_features(self, context_obj):
    if not context_obj.has_key('target_pos'):
      context_obj['target_pos'] = self._call_tagger(context_obj['target'])
    if not context_obj.has_key('source_pos') and context_obj.has_key['source']:
      context_obj['source_pos'] = self._call_tagger(context_obj['source'], lang='src')

    #extract POS features:
    # - target POS
    # - source POS (may be more than 1)
    # - something else?
    tg_pos = context_obj['target_pos'][context_obj['index']]
    src_pos = ''
    if context_obj.has_key('source_pos') and context_obj.has_key('alignments'):
      src_pos = [ context_obj['source_pos'][i] for i in context_obj['alignments'][context_obj['index']] ]

    return (tg_pos, src_pos)
