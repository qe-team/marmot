#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import os, re
from subprocess import call
from marmot.preprocessing.get_double_corpus import get_double_string, get_double_corpus


class GetDoubleCorpusTests(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.dirname(os.path.realpath(__file__))
        self.align = os.path.join(self.test_dir, 'test_data/alignments/test.de-en.gdfa')
        self.one = os.path.join(self.test_dir, 'test_data/alignments/test.de-en')
        self.two = (os.path.join(self.test_dir, 'test_data/alignments/test.de'), os.path.join(self.test_dir, 'test_data/alignments/test.en'))
        self.test_str = u'three_drei in_zehn ten_zehn south_südafrikanern africans_jünger are_sind younger_15 than_als 15_das ,_, meaning_, that_dass they_sie did_UNALIGNED not_nicht live_tag a_der day_apartheid under_gelebt apartheid_haben ._.'

    # remove the tmp_* files created by the tests
    def tearDown(self):
        for f in os.listdir(self.test_dir):
            if re.search("^tmp_*", f):
                os.remove(os.path.join(self.test_dir, f))

    def test_get_double_string_wrong(self):
        get_double_string( ['hallo','welt'], ['hello',',','world'],'0-0 0-1 1-2 2-2' )

    def test_get_double_string_right(self):
        all_str='Unsere Einblicke ins All : Die wichtigsten Teleskope ||| Our insights in all : The most important telescopes'
        src = all_str[:all_str.find('|||')].strip().split()
        trg = all_str[all_str.find('|||')+3:].strip().split()
        align = '0-0 1-1 2-2 3-3 4-4 5-5 6-6 6-7 7-8'
        new_line = get_double_string(src, trg, align)
        self.assertEqual(new_line, ['Our_Unsere', 'insights_Einblicke', 'in_ins', 'all_All', ':_:', 'The_Die', 'most_wichtigsten', 'important_wichtigsten', 'telescopes_Teleskope'])

    def test_get_double_corpus_one(self):
        alignment_file = os.path.join(self.test_dir, 'test_data/alignments/test.de-en.gdfa.double')
        if os.path.isfile(alignment_file):
            call(['rm', alignment_file])
        get_double_corpus(self.align, one_file=self.one)
        a_str = open(alignment_file).readline()[:-1].decode('utf-8')
        self.assertEqual(a_str, self.test_str)

    def test_get_double_corpus_two(self):
        alignment_file = os.path.join(self.test_dir, 'test_data/alignments/test.de-en.gdfa.double')
        if os.path.isfile(alignment_file):
            call(['rm', alignment_file])
        get_double_corpus(self.align, two_files=self.two)
        a_str = open(alignment_file).readline()[:-1].decode('utf-8')
        self.assertEqual(a_str, self.test_str)



if __name__ == '__main__':
    unittest.main()
