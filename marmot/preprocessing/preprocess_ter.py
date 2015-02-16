import sys
import numpy as np

def parse_hyp_loc_map( line ):
  numbers = [int(x) for x in line.split()]
  orig2shifted = { i:j for (j,i) in list(enumerate(numbers)) }
  shifted2orig = dict( enumerate(numbers) )
  return ( orig2shifted, shifted2orig )

def parse_sentence( line_array ):
  hyp, ref = [], []
  orig2shifted, shifted2orig = {}, {}
  align, sentence_id = "",""
  for line in line_array:
    line_separator = line.find(':')
    line_id = line[:line_separator]
#    print "New line: ", line_id
    if line_id == "Hypothesis":
      hyp = [w for w in line[line_separator+2:-1].split()]
    elif line_id == "Reference":
      ref = [w for w in line[line_separator+2:-1].split()]
    elif line_id == "Sentence ID":
      sentence_id = line[line_separator+2:-1]
    elif line_id == "Alignment":
      align = line[line_separator+3:-2]
      # 'Deletion' errors are not considered as there is no word in hypothesis for them
      align = align.replace('D','')
      print('parse sentence - align: ')
      print(align)
    elif line_id == "HypLocMap":
      ( orig2shifted, shifted2orig ) = parse_hyp_loc_map( line[line_separator+2:-1] )
    else: continue
  hyp = np.array( hyp, dtype=object )
  ref = np.array( hyp, dtype=object )
  return ( sentence_id, hyp, ref, orig2shifted, align )

def get_features( sentence_id, sentence, labels, good_context ):
  good_label = u'GOOD'
  bad_label = u'BAD'
#  print "Sentence: ", sentence
#  print "Labels: ", labels

  assert( len(sentence) == len(labels) )

  instances = []

  for i in range(len(labels)):
    prev_word, next_word = "",""
    good_left, good_right = False, False
    if i == 0: 
      prev_word = u"START"
      good_left = True
    else: prev_word = sentence[i-1]
    if i+1 == len(labels): 
      next_word = u"END"
      good_right = True
    else: next_word = sentence[i+1]

    if not good_left:
      good_left = ( not good_context or labels[i-1] == 'G' )
    if not good_right:
      good_right = ( not good_context or labels[i+1] == 'G' )

    if good_left and good_right:
      cur_label = good_label if labels[i] == 'G' else bad_label
      instances.append( np.array([ sentence_id, i, sentence[i], prev_word, next_word, sentence, cur_label, cur_label ]) )
  return np.array( instances )
    
#output format: array of training instances
#each instance is an array of:
#   sentence id, word id, word_i, word_i-1, word_i+1, sentence, label, label
#label appears twice for compatibility with fine-grained error classification
def parse_ter_file( pra_file_name, good_context=True ):
  a_file = open( pra_file_name )
  sys.stderr.write("Parse file \'%s\'\n" % pra_file_name)
  features = []
  cur_sentence = []
  for line in a_file:
#    print line
    cur_sentence.append( line.decode("utf-8") )
    if line.startswith('Score: '):
      ( sent_id, hyp, ref, orig2shifted, align ) = parse_sentence( cur_sentence )
      if len( hyp ) != len( align ):
        sys.stderr.write("Hypothesis and alignment map don't match, sentence number %s\n" % sent_id)
        cur_sentence = []
        continue

      err_labels = ""
      for i in range(len(hyp)):
        if align[orig2shifted[i]] == ' ':
          err_labels += 'G'
        elif align[orig2shifted[i]] == 'S' or align[orig2shifted[i]] == 'I':
          err_labels += 'B'
      features.extend( get_features( sent_id, hyp, err_labels, good_context ) )

      cur_sentence = []
  return np.array( features, dtype=object )
