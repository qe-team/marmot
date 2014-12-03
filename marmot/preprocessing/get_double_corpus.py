import sys


def get_double_string( words_src, words_trg, align_str, cnt=0 ):
  '''
  Generation of a line of double tokens.

  <src_list> -- list of source tokens
  <trg_list> -- list of target tokens
  <align_str> -- string with alignments in format "i-j" (source-target)
  
  Returns: list of double tokens (target_source in target word order)
  '''

 # align_pairs = align_str[:-1].decode('utf-8').split()
  alignment = {int(i):int(j) for (j,i) in [tuple(pair.split('-')) for pair in align_str.split()]}

  default = u'UNALIGNED'
  new_string = []
  try:
    if max(alignment.keys()) >= len(words_trg):
      raise IndexError('Too few words in target at line', cnt, max(alignment.keys()), len(words_trg))
    if max(alignment.values()) >= len(words_src):
      raise IndexError('Too few words in source at line', cnt, max(alignment.keys()), len(words_src))

    for i in range(len(words_trg)):
      new_token = u''
      if alignment.has_key(i):
        new_token = words_trg[i]+u'_'+words_src[alignment[i]]
      else:
        new_token = words_trg[i]+u'_'+default
      new_string.append(new_token)

  #one of word indices is larger than sentence length
  except IndexError as e:
    sys.stderr.write("%s %d: %d words, %d required\n" % (e.args[0], e.args[1], e.args[3], e.args[2]+1 ))

  finally:
    return new_string


def get_double_corpus( align, two_files=("",""), one_file="", aligned_file=""):
  """
  Get corpus that consists of target_source tokens in target word order
  <align> -- alignments in i-j format
  <two_files>: pair of files (source, target)
  <one_file>: single bilingual file where each string is source_sentence ||| target_sentence
  the new corpus is saved to file <aligned_file>
  """
  if two_files[0] and two_files[1]:
    one = False
  elif one_file:
    one = True
  else:
    sys.stderr.write("No text file provided\n")
    return

  if not aligned_file:
    aligned_file = align+'.double'

  f_align = open(align)
  if one:
    f_double_src = open(one_file)
  else:
    f_src = open(two_files[0])
    f_trg = open(two_files[1])

  f_out = open( aligned_file, 'w' )
  
  cnt = 0
  for l_align in f_align:
    if one:
      line = f_double_src.readline()
      if not line: break
      if line.find('|||') == -1:
        sys.stderr.write("Wrong text file format\n")
        break
      src = line[:line.find('|||')]
      trg = line[line.find('|||')+4:]
    else:
      src = f_src.readline()
      trg = f_trg.readline()
      if not src or not trg:
        sys.stderr.write("Lengths of text files don't match\n")
        break

    words_src = src[:-1].decode('utf-8').strip().split()
    words_trg = trg[:-1].decode('utf-8').strip().split()
 
    new_string = get_double_string( words_src, words_trg, l_align, cnt )
    cnt += 1
    f_out.write("%s\n" % (' '.join( [w.encode('utf-8') for w in new_string] )))

  f_out.close()
  f_align.close()
  if one:
    f_double_src.close()
  else:
    f_src.close()
    f_trg.close()
