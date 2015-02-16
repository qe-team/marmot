# -*- coding: utf-8 -*-

import sys
from xml.dom.minidom import parseString
from string import punctuation
import numpy as np
from subprocess import Popen, PIPE, STDOUT
import os, codecs
from collections import defaultdict

cdec_home = ""

class Correction:
  def __init__(self, _start, _end, _type, _id):
    self.start = _start
    self.end = _end
    self.type = _type
    self.id = _id

#parse sentence
#line - string from the file, contains sentence id, source sentence and target with error markup
def parse_line( line ):
  global cdec_home
  if line[:-1] == '\n':
    line = line[:-1]
  line = line.decode('utf-8')
  chunks = line.split('\t')
  if np.size(chunks) != 3:
    sys.stderr.write("Wrong format\n")
    return("","",[],[])
  sentence_id = chunks[0]#.decode("utf-8")
  src = chunks[1]#.decode("utf-8")
  trg = []
  corrections = []
    
  annotation = '<?xml version="1.0" encoding="utf-8"?><mqm:translation xmlns:mqm=\"MQM\">'+chunks[2].encode('utf-8')+'</mqm:translation>'
  try:
    sentence = parseString( annotation )
  # sentence = parseString( annotation )
  # TODO: what is the error here and why does it happen?
  except UnicodeEncodeError as e:
    sys.stderr.write("Sentence \'%s\' not parsed\n" % sentence_id)
    print(e)
    print(annotation)
    return ("", "", [], [])
  except:
    print(sys.exc_info()[0])
    print(annotation)
    return("", "", [], [])

  if not "CDEC_HOME" in os.environ:
    cdec_home='/home/varvara/software/cdec'
    sys.stderr.write("$CDEC_HOME variable not specified, using %s\n" % cdec_home)
  else:
    cdec_home = os.environ['CDEC_HOME']

  #tokenize source sentence
  FNULL = open(os.devnull, 'w')
  p = Popen([cdec_home+"/corpus/tokenize-anything.sh"], stdout=PIPE, stdin=PIPE, stderr=FNULL)
  tok = p.communicate(input=src.encode('utf-8'))[0].strip()
  src = tok.decode('utf-8')
  FNULL.close()

  curr_word = 0
  opened_issues = {}

  #parse sentence xml
  for elem in sentence.documentElement.childNodes:
    #element
    if elem.nodeType == 1:
      try:
        el_id = int(elem.attributes["id"].value)
        if elem.nodeName == "mqm:startIssue":
          opened_issues[el_id] = ( curr_word, elem.attributes["type"].value )

        elif elem.nodeName == "mqm:endIssue":
          if not opened_issues.has_key( el_id ):
            sys.stderr.write( "Inconsistent error %d\n" % el_id )
            return ("", "", [], [])

          a_corr = Correction( opened_issues[el_id][0], curr_word, opened_issues[el_id][1], el_id )
          corrections.append( a_corr )
          del opened_issues[el_id]
      #some element attributes can be missing
      except KeyError as e:
        sys.stderr.write("Missing attribute in sentence %s: %s\n" % (sentence_id, e.args[0]))
        return("", "", [], [])
      except:
        sys.stderr.write(sys.exc_info())
        return("", "", [], [])

    #text
    elif elem.nodeType == 3:

      FNULL = open(os.devnull, 'w')
      p = Popen([cdec_home+"/corpus/tokenize-anything.sh"], stdout=PIPE, stdin=PIPE, stderr=FNULL)
      tok = p.communicate(input=elem.nodeValue.encode("utf-8"))[0].strip()
      FNULL.close()
      words = [w.decode('utf-8') for w in tok.split()]
      trg.extend( words )
      curr_word += len( words )

  if len( opened_issues ):
    sys.stderr.write( "Inconsistent error(s): %s\n" % ( ', '.join( [str(x) for x in opened_issues.keys()] ) ) )
    return ("", "", [], [])

  return ( sentence_id, src, np.array(trg, dtype=object), np.array(corrections,dtype=object) )

#parse file 'file_name' and write result to 'out_file' (doesn't write if 'out_file'=='')
#if good_context==True extract only words whose contexts (w[i-1], w[i+1]) are labelled 'GOOD'
#
#return an array of errors
#every error is array = [sentence_id, word_index, w[i], w[i-1], w[i+1], sentence, label]
#    word_index - integer
#    sentence - array of unicode strings
#    sentence_id, w[i], w[i-1], w[i+1], label - unicode strings
#
def parse_src( file_name, good_context=True, out_file="" ):
  global cdec_home
  cdec_home = os.environ['CDEC_HOME']
  if not cdec_home:
    sys.stderr.write('Cdec decoder not installed or CDEC_HOME variable not set\n')
    sys.stderr.write("Please set CDEC_HOME variable so that $CDEC_HOME/corpus directory contains \'tokenize-anything.sh\'\n")
    return ("","",[],[])

  f_src = open(file_name)
  sys.stderr.write("Parsing file \'%s\n" % (file_name))
  instances = []
  for line in f_src:
    ( sentence_id, src, trg, corrections ) =  parse_line( line )
    if not sentence_id: continue 
    instances.extend( get_instances( sentence_id, src, trg, corrections, good_context ) )

  f_src.close()

  if out_file:
    f_out = open( out_file, 'w' )
    for ii in instances:
      f_out.write( "%s\t%d\t%s\t%s\t%s\t%s\t%s\n" % ( ii[0].encode("utf-8"), ii[1], ii[2].encode("utf-8"), ii[3].encode("utf-8"), ii[4].encode("utf-8"), ' '.join( [s.encode("utf-8") for s in ii[5]] ), ii[6] ))
    f_out.close()

  return np.array( instances )

#output format: sentence_id, word_index, word_i, word_i-1, word_i+1, sentence, binary_label, error_type
def get_instances(sentence_id, src, trg, corrections, good_context):
  good_label = u'GOOD'
  bad_label = u'BAD'
 
  instances = []

  word_errors = [ [] for i in range(len(trg)) ]
  for err in corrections:
    for i in range( err.start, err.end ):
      word_errors[i].append( ( err.id, err.type ) )

  for i in range(len(trg)):
    first, last = False, False
    if i == 0: first = True
    if i + 1 == len(trg): last = True

    #check if contexts contain errors
    #if check not needed, set good_context=False and good_left and good_right will always be True
    good_left = ( first or not len( word_errors[i-1] ) or not good_context )
    good_right = ( last or not good_context )
    if not good_right: good_right = ( not len(word_errors[i+1]) )
    if not good_left or not good_right: 
      continue

    if last: next_word = u'END'
    else: next_word = trg[i+1]
    if first: prev_word = u'START'
    else: prev_word = trg[i-1]

    if not len( word_errors[i] ):
      instances.append( [sentence_id, i, trg[i], prev_word, next_word, trg, good_label, u'OK'] )
    elif len( word_errors[i] ) == 1:
      instances.append( [sentence_id, i, trg[i], prev_word, next_word, trg, bad_label, word_errors[i][0][1]] )

  return np.array( instances, dtype=object )

# Convert WMT data in xml into 2 formats:
#   -- plain text (only automatic translation, no error markup)
#   -- word<TAB>label
# New files are saved to the directory of the source file ('file_name') with extensions 'txt' and 'words'
def convert( file_name ):

  f_xml = open( file_name )
  prefix = file_name[:file_name.rfind('.')]
  f_plain = open( prefix+'.txt', 'w' )
  f_words = open( prefix+'.words', 'w' )
  for line in f_xml:
    ( sentence_id, src, trg, corrections ) = parse_line( line )
    
    # if xml is not parsed
    if not sentence_id:
      continue

    trg_label = [ u'GOOD' for w in trg ]
    for c in corrections:
      for i in range( c.start, c.end ):
        trg_label[i] = u'BAD'
    f_plain.write( "%s\n" % (' '.join( trg )) )
    for i in range(len(trg)):
      f_words.write( "%s\t%s\n" % ( trg[i].encode('utf-8'), trg_label[i].encode('utf-8') ) )

  f_xml.close()
  f_plain.close()
  f_words.close()

#Convert WMT data to file accepted by fast_align: source ||| target
#only automatic translation, no error markup
def convert_to_double_file( file_name ):
  f_xml = open( file_name )
  # TODO: there is a bad bug here
  f_double_name = file_name[:file_name.rfind('.')]+'.double'
  f_double = open( f_double_name, 'w' )
  for line in f_xml:
    ( sentence_id, src, trg, corrections ) = parse_line( line )

    # if xml is not parsed
    if not sentence_id:
      f_double.write('\n')
      continue

    f_double.write( "%s ||| %s\n" % (src.encode('utf-8'), ' '.join( [ii.encode('utf-8') for ii in trg] )) )

  f_xml.close()
  f_double.close()
  # if file_name.rfind('.') == -1
  return f_double_name

#return 3 labels for every word: fine-grained, coarse-grained (fluency/adequacy/good) and binary
#write to file in WMT gold standard format:
#     sentence_num<TAB>word_num<TAB>word<TAB>fine_label<TAB>coarse_label<TAB>binary_label
def get_all_labels( sentence_id, trg, corrections ):
  label_fine = [u'OK' for w in trg]
  label_coarse = [u'OK' for w in trg]
  label_bin = [u'OK' for w in trg]
 
  #mapping between coarse and fine-grained labels
  #unknown error is aliased as 'Fluency'
  coarse_map = defaultdict(lambda: u'Fluency')

  for w in ['Terminology','Mistranslation','Omission','Addition','Untranslated','Accuracy']:
    coarse_map[w] = u'Accuracy'
  for w in ['Style/register','Capitalization','Spelling','Punctuation','Typography','Morphology_(word_form)','Part_of_speech','Agreement','Word_order','Function_words','Tense/aspect/mood','Grammar','Unintelligible','Fluency']:
   coarse_map[w] = u'Fluency'

  for c in corrections:
    for i in range(c.start,c.end):
      label_fine[i] = c.type
      label_coarse[i] = coarse_map[c.type]
      label_bin[i] = u'BAD'

  out = []
  for i in range(len(trg)):
    try:
      out.append(u'\t'.join([sentence_id, unicode(i), trg[i], label_fine[i], label_coarse[i], label_bin[i]]))
    except IndexError as e:
     print str(i)+"!!!!", len(trg), len(label_fine), len(label_coarse), len(label_bin)

  return out


# convert in parallel
if __name__ == '__main__':
    import multiprocessing
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,  help='training data file -- .tsv with training data')

    args = parser.parse_args()
    file_name = args.input

    prefix = file_name[:file_name.rfind('.')]
    f_plain = open(prefix+'.txt', 'w')
    f_words = open(prefix+'.words', 'w')

    f_xml = open(file_name)
    # ( sentence_id, src, trg, corrections ) = parse_line( line )
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    result = pool.map_async(parse_line, f_xml)
    parsed_lines = result.get()

    for l in parsed_lines:
        (sentence_id, src, trg, corrections) = l
        # if xml is not parsed
        if not sentence_id:
            continue

        trg_label = [u'GOOD' for w in trg]
        for c in corrections:
            for i in range(c.start, c.end):
                trg_label[i] = u'BAD'
        f_plain.write("%s\n" % (' '.join(trg)))
        for i in range(len(trg)):
            f_words.write("%s\t%s\n" % (trg[i].encode('utf-8'), trg_label[i].encode('utf-8')))

    f_xml.close()
    f_plain.close()
    f_words.close()
