#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest, os, tempfile, sys
import glob
from subprocess import call

from marmot.preprocessing.parsers import *
from marmot.util.simple_corpus import SimpleCorpus


class BackTransParseTests(unittest.TestCase):
    def setUp(self):
        self.interesting_tokens = set(['the','it'])
        module_path = os.path.dirname(__file__)
        self.corpus_file = os.path.join(module_path, 'test_data/negative_data/back_translation.txt')
        self.labels_file = os.path.join(module_path, 'test_data/negative_data/back_translation.labels')

    def test_parse_back_translation(self):
        token_contexts = parse_back_translation(self.corpus_file, self.labels_file, set(['the']))
        self.assertTrue(len(token_contexts) > 0)
        no_token_contexts = parse_back_translation(self.corpus_file, self.labels_file, set(['egraweaweg']))
        self.assertTrue(len(no_token_contexts) == 0)


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

class TestAdditionalRepresentations(unittest.TestCase):

    def setUp(self):
        self.interesting_tokens = set(['the','it'])
        module_path = os.path.dirname(__file__)
        self.tg_corpus_path = os.path.join(module_path, 'test_data/corpus.en.1000')
        self.src_corpus_path = os.path.join(module_path, 'test_data/corpus.de.1000')
        self.source = SimpleCorpus(self.src_corpus_path)
        self.target = SimpleCorpus(self.tg_corpus_path)
        tagger_path = os.environ['TREE_TAGGER'] if os.environ.has_key('TREE_TAGGER') else ''
        if tagger_path == '':
            sys.stderr.write('TreeTagger is not installed or TREE_TAGGER variable is not set\n')
        self.tagger = tagger_path+'/bin/tree-tagger'
        self.parameter_file = tagger_path+'/lib/english-utf8.par'


    def test_create_new_instance_additional(self):
        obj = create_new_instance_additional('boy', 1, target=['a','boy','hits','a','dog'], label=1, elements=[['DT','NN','VVZ','DT','NN'], [u'un', u'garçon', u'frappe', u'un', u'chien']], elem_labels=['target_pos', 'source'])
        self.assertTrue( obj['target_pos'] == ['DT','NN','VVZ','DT','NN'] )
        self.assertTrue( obj['source'] == [u'un', u'garçon', u'frappe', u'un', u'chien'] )


    def test_parse_corpus_contexts_additional(self):

        # tagging
        (label_tag, tag_list) = get_pos_tagging( self.tg_corpus_path, self.tagger, self.parameter_file, 'target_pos' )
        self.assertTrue( label_tag == 'target_pos' )
        for sent, tags in zip(self.target.get_texts(), tag_list):
            self.assertTrue( len(sent) == len(tags) )

        for a_file in glob.glob('tmp_final*'):
            call(['rm', a_file])

        # alignment
        (label_align, alignments) = get_alignments(self.src_corpus_path, self.tg_corpus_path, trained_model = None, src_train=self.src_corpus_path, tg_train=self.tg_corpus_path, align_model = 'align_model', label='alignments')
        self.assertTrue(label_align == 'alignments')
        for src, tg, align in zip(open(self.src_corpus_path), open(self.tg_corpus_path), alignments):
            src_words = src[:-1].split()
            tg_words = tg[:-1].split()
            self.assertTrue(len(align) == len(tg_words))
            self.assertTrue( [num < len(src_words) for a in align for num in a] )

        for a_file in glob.glob('align_model.*'):
            call(['rm', a_file])
        for a_file in glob.glob(os.path.basename(self.src_corpus_path)+'_'+os.path.basename(self.tg_corpus_path)+'*'):
            call(['rm', a_file])

        additional = [(label_tag, tag_list), ('source', self.source.get_texts()), (label_align, alignments)]
        out=parse_corpus_contexts_additional(self.tg_corpus_path, self.interesting_tokens, 1, *additional)
  

if __name__ == '__main__':
    unittest.main()

