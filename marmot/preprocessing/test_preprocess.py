# -*- coding: utf-8 -*-

import unittest
import sys
import StringIO
import numpy as np

import preprocess

class TestPreprocess(unittest.TestCase):

  def test_wrong_format(self):
    a_stream = StringIO.StringIO()
    sys.stderr = a_stream
    self.assertTrue( preprocess.parse_line( "This is not a valid format" ) == ("","",[],[]))
    self.assertTrue( a_stream.getvalue() == "Wrong format\n" )
    a_stream.close()
  
  def test_inconsistent_issues(self):
    a_stream = StringIO.StringIO()
    sys.stderr = a_stream
    self.assertTrue( preprocess.parse_line( """z/2012/12/01/198819-18_de_MT2\tDas Wort orientiert sich an "Bang Dakuan".\tThe word <mqm:startIssue type="Terminology" severity="critical" note="" agent="annot16" id="4279"/>orientates<mqm:endIssue id="4279"/> itself <mqm:startIssue type="Function words" severity="critical" note="" agent="annot16" id="4280"/>by "Bang Dakuan".""" ) == ("","",[],[]) )
    self.assertTrue( preprocess.parse_line( """derstandart.at/2012/12/01/141907-37_de_MT2\tDas ist billig und zeiteffizient.\tThis is cheap and time-efficient<mqm:endIssue id="4281"/>.""") == ("","",[],[]) )
#    self.assertTrue( a_stream.getvalue() == "Inconsistent error(s): 4280\nInconsistent error 4281\n" )
    print 'inconsistent issues: ', a_stream.getvalue()
    a_stream.close()

  def test_invalid_xml(self):
    a_stream = StringIO.StringIO()
    sys.stderr = a_stream
    self.assertTrue( preprocess.parse_line("""faz/2012/12/01/198819-55_de_MT2\tMan muss höllisch aufpassen\tOne must pay attention <mqm:startIssue type="Mistranslation" severity="critical" note="" agent="annot16" id="4290"/ >like hell<mqm:endIssue id="4290"/>""" ) == ("","",[],[]) )
#    self.assertTrue( a_stream.getvalue() == "Sentence \'faz/2012/12/01/198819-55_de_MT2\' not parsed\n" )
    print 'invalid xml: ', a_stream.getvalue()
    a_stream.close()

  def test_cyrillic_str(self):
#    preprocess.parse_line( open('test_data.txt').readline()[:-1] )
    preprocess.parse_line( """z/2012/12/01/198819-18_de_MT2\tDas Wort orientiert sich an "Bang Dakuan".\tФарш невозможно <mqm:startIssue type="Terminology" severity="critical" note="" agent="annot16" id="4279"/>провернуть<mqm:endIssue id="4279"/> назад <mqm:startIssue type="Function words" severity="critical" note="" agent="annot16" id="4280"/>и<mqm:endIssue id="4280"/> мясо из котлет не востановишь.""" )

  def test_tokenizer(self):
    (a, b, a_list, aa_list) = preprocess.parse_line( """z/2012/12/01/198819-18_de_MT2\tDas Wort orientiert sich an "Bang Dakuan".\tThis, sentence, very-very <mqm:startIssue type="Terminology" severity="critical" note="" agent="annot16" id="4279"/>http://website.com (complicated)<mqm:endIssue id="4279"/> 10,000 and 0.1: to tokenize; don't <mqm:startIssue type="Function words" severity="critical" note="" agent="annot16" id="4280"/>and wouldn't e.g.<mqm:endIssue id="4280"/> he'll "many" John's $200 other things.""" )
    self.assertTrue( np.array_equal( a_list, [u'This', u',', u'sentence', u',', u'very', u'-', u'very', u'http://website.com', u'(', u'complicated', u')', u'10,000', u'and', u'0.1', u':', u'to', u'tokenize', u';', u'do', u"n't", u'and', u'would', u"n't", u'e.g.', u'he', u"'ll", u'"', u'many', u'"', u'John', u"'s", u'$', u'200', u'other', u'things',u'.'] ) )
    self.assertTrue( aa_list[0].start == 7 and aa_list[0].end == 11 )
    self.assertTrue( aa_list[1].start == 20 and aa_list[1].end == 24 )
#    for err in aa_list:
#      sys.stdout.write("Error %s: from %d to %d: \'%s\'\n" % (err.id, err.start, err.end, ' '.join(a_list[err.start:err.end]) ))


if __name__ == "__main__":
  unittest.main()
