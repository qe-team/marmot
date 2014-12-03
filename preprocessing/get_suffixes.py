import sys
from collections import defaultdict
from gensim import corpora


# find the longest suffix the word contains
def find_suffix( word, suffix_list, prefix=u'__' ):
  #start searching from the longest suffixes (length of word - 2)
  for i in range( min(max(suffix_list.keys()),len(word)-2), min(suffix_list.keys())-1, -1 ):
    for s in suffix_list[i]:
      if word.endswith(s):
        return prefix+s
      #plural nouns 
      elif word.endswith(s+u's'):
        return prefix+s+u's'
      elif word.endswith(s+u'es'):
        return prefix+s+u'es'

  return word

# suffix list - dictionary: {<suffix_length>:[list of suffixes of this length]}
def form_suffix_list( suffix_file ):
  suffix_list = defaultdict(lambda: set())
  for line in open( suffix_file ):
    suffix = line[:-1].decode('utf-8')
    suffix_list[len(suffix)].add( suffix )

  return suffix_list



def get_suffixes( txt_file, suffix_file, stdout_file="", threshold=sys.maxint, prefix=u'__' ):
  """
  Replace all words in <txt_file> with suffixes where possible.
  Set of suffixes must be provided in <suffix_file>
  The new corpus is written to <stdout_file> or to standard output if no file provided
  <prefix> -- string to replace the non-suffix part of the word (default '__': information -> __tion)  

  Words are replaced with suffixes only if occurred in corpus less times than <threshold>
  Default: no threshold (all words replaced)
  """

  out = open( stdout_file, 'w' ) if stdout_file else sys.stdout 
  sys.stderr.write('Loading corpus\n') 
  my_corp = corpora.TextCorpus(txt_file)
  sys.stderr.write('Building suffix list\n')
  suffix_list = form_suffix_list(suffix_file)
  sys.stderr.write('Suffix search\n')

  #replace only words that occur in corpus less times than threshold
  #default - no threshold (all words are replaced with suffix)
  dict_copy = dict( [ (token,find_suffix(token, suffix_list, prefix=prefix)) if my_corp.dictionary.dfs[id] < threshold else (token,token) for (id, token) in my_corp.dictionary.items() ] )
  print dict_copy
  sys.stderr.write('Output\n')
 
  cnt = 0
  
  in_file = open(txt_file)
  for line in in_file:
    cnt += 1
    if cnt%10000 == 0:
      sys.stderr.write('.')
    words = line[:-1].decode('utf-8').split()
    for w in words:
      try:
        out.write("%s " % dict_copy[w].encode('utf-8'))
      except KeyError:
        dict_copy[w] = w
        out.write("%s " % dict_copy[w].encode('utf-8'))
    out.write("\n")
  in_file.close()

  if stdout_file: out.close()
