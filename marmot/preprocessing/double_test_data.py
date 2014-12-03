import sys, os
from subprocess import check_call
import argparse
from collections import defaultdict

from parse_xml import parse_line
from get_double_corpus import get_double_corpus

#naming
#input WMT - <wmt>
#source ||| target - <wmt>.src_trg
#alignments - <wmt>.gdfa
#token-aligned file - <wmt>.double
#one word per line (test format) - <wmt>.words


#return 3 labels for every word: fine-grained, coarse-grained (fluency/adequacy/good) and binary
#write to file in WMT gold standard format:
#     sentence_num<TAB>word_num<TAB>word<TAB>fine_label<TAB>coarse_label<TAB>binary_label
def get_all_labels( sentence_num, trg, corrections ):
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
    out.append(u'\t'.join([unicode(sentence_num)+u'.1', unicode(i), trg[i], label_fine[i], label_coarse[i], label_bin[i]]))

  return out


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('wmt', help="WMT data in XMl format")
  parser.add_argument('align', help="alignment model location")
  args = parser.parse_args()

  file_xml_name = args.wmt
  align_model = args.align

  file_src_trg_name = file_xml_name+'.src_trg'
  file_src_trg_lc_name = file_xml_name+'.src_trg.lc'
  file_alignments_name = file_xml_name+'.gdfa'
  file_double_name = file_xml_name+'.double'
  file_words_name = file_xml_name+'.words'
  

  cdec_home = os.environ['CDEC_HOME']
  if cdec_home == "":
    sys.stderr.write("No CDEC_HOME variable found. Please install cdec and/or set the CDEC_HOME variable to cdec root directory\n")
    sys.exit(2)

  file_xml = open( file_xml_name )
  file_src_trg = open( file_src_trg_name, 'w' )
  file_words = open( file_words_name, 'w' )

  cur_sentence = 0
  sentence_map = {}
  
  sys.stderr.write("Parsing xml\n")
  for line in file_xml:
    cur_sentence += 1
    if cur_sentence%10 == 0:
      sys.stderr.write('.')
    ( sentence_id, src, trg, corrections ) = parse_line( line )

    # if xml is not parsed
    if not sentence_id:
      sys.stderr.write("Sentence %d not parsed\n" % cur_sentence)
      sentence_map[cur_sentence] = u'NOT_PARSED'
      continue
    sentence_map[cur_sentence] = sentence_id
    out = get_all_labels( cur_sentence, trg, corrections )
    file_words.write("%s\n" % ('\n'.join([a.encode('utf-8') for a in out])))

    file_src_trg.write( "%s ||| %s\n" % (src.encode('utf-8'), ' '.join( [ii.encode('utf-8') for ii in trg] )) )

  file_xml.close()
  file_src_trg.close()
  file_words.close()

  #lowercase
  file_src_trg = open( file_src_trg_name )
  file_src_trg_lc = open(file_src_trg_lc_name, 'w')
  sys.stderr.write("\nForce aligning\n")
  check_call([ cdec_home+'/corpus/lowercase.pl' ], stdin=file_src_trg, stdout=file_src_trg_lc)
  file_src_trg_lc.close()
  file_src_trg.close()

  #force align
  file_src_trg_lc = open( file_src_trg_lc_name )
  file_alignments = open( file_alignments_name, 'w' )
  check_call([ cdec_home+'/word-aligner/force_align.py', align_model+'.fwd_params',  align_model+'.fwd_err', \
             align_model+'.rev_params', align_model+'.rev_err' ], stdin=file_src_trg_lc, stdout=file_alignments)
  file_src_trg_lc.close()
  file_alignments.close()
 
  get_double_corpus( file_alignments_name, one_file=file_src_trg_lc_name, aligned_file=file_double_name)

