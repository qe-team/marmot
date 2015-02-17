#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest, os, tempfile, sys
import glob

from marmot.parsers.parsers import *
from marmot.util.simple_corpus import SimpleCorpus


class TestCorpusParser(unittest.TestCase):
    def setUp(self):
        self.interesting_tokens = set(['the','it'])
        module_path = os.path.dirname(__file__)
        self.corpus_path = os.path.join(module_path, 'test_data/corpus.en.1000')
        self.corpus = SimpleCorpus(self.corpus_path)

    def test_parse_corpus_contexts(self):
        contexts = parse_corpus_contexts(self.corpus_path, self.interesting_tokens)
        for context in contexts:
            self.assertTrue(len(set(context['target']).intersection(self.interesting_tokens)) > 0)
        all_contexts = parse_corpus_contexts(self.corpus_path)
        num_toks = sum([len(sen) for sen in self.corpus.get_texts()])
        self.assertTrue(num_toks == len(all_contexts))

class TestImportantTokens(unittest.TestCase):

    def setUp(self):
        self.interesting_tokens = set(['the','it'])
        module_path = os.path.dirname(__file__)
        self.corpus_path = os.path.join(module_path, 'test_data/wmt.en.1000')

    def test_extract_important_tokens(self):
        contexts = [['this', 'is', 'a', 'test'], ['This', 'is', 'another', 'test', '.']]
        temp = tempfile.NamedTemporaryFile(delete=False)
        for l in contexts:
            temp.write((' ').join(l) + '\n')
        temp.close()

        important_tokens = extract_important_tokens(temp.name, min_count=2)
        self.assertTrue('is' in important_tokens)
        self.assertFalse('This' in important_tokens)


class TestParseWMT(unittest.TestCase):

    def setUp(self):
        self.interesting_tokens = set(['the','it'])
        module_path = os.path.dirname(__file__)
        self.corpus_path = os.path.join(module_path, 'test_data/DE_EN.tgt_ann.test')
        self.source_path = os.path.join(module_path, 'test_data/DE_EN.source.test')

    def test_parse_wmt14_data_no_source(self):
        contexts = parse_wmt14_data(self.corpus_path, '')
        for context in contexts:
            self.assertTrue(context['token'] == context['target'][context['index']])

    def test_parse_wmt14_data(self):
        contexts = parse_wmt14_data(self.corpus_path, self.source_path)
        for context in contexts:
            self.assertTrue(context['token'] == context['target'][context['index']])
            self.assertTrue(context.has_key('source'))


class TestSemevalParser(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(__file__)
        self.inputfile = os.path.join(module_path, 'test_data/semeval/STS.input.MSRvid.txt')
        self.scoresfile = os.path.join(module_path, 'test_data/semeval/STS.gs.MSRvid.txt')

    def test_parse_semeval(self):
        contexts = parse_semeval(self.inputfile, self.scoresfile)[:10]
        for context in contexts:
            self.assertTrue('source' in context and 'target' in context)
            self.assertTrue(type(context['source']) == list and type(context['target']) == list)
            self.assertTrue(len(context['source']) > 0 and len(context['target']) > 0)


if __name__ == '__main__':
    unittest.main()

