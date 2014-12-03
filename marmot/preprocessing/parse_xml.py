import sys
from xml.dom.minidom import parseString
import numpy as np
from subprocess import Popen, PIPE
import os

class Correction:
  def __init__(self, _start, _end, _type, _id):
    self.start = _start
    self.end = _end
    self.type = _type.replace(' ','_')
    self.id = _id

def parse_line( line ):
  '''parse a sentence with xml markup
     line - string from the file, contains the tab-separated sentence id, source sentence and target with error markup
  '''
  global cdec_home
  line = line[:-1].decode('utf-8')
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


