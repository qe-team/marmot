#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys
import unittest
import StringIO

from marmot.features.pos_feature_extractor import POSFeatureExtractor

class POSFeatureExtractorTests(unittest.TestCase):

  # check: POS rerpresentation in context_obj
  # no POS representation
    def setUp(self):
        tagger_root = os.environ['TREE_TAGGER'] if os.environ.has_key('TREE_TAGGER') else ''
        if tagger_root == '':
            sys.stderr('TREE_TAGGER environment variable should be defined so that $TREE_TAGGER/bin/tree-tagger exists\n')
            sys.exit(2)
        self.tagger = tagger_root+'/bin/tree-tagger'
        self.par_src = tagger_root+'/lib/english-utf8.par'
        module_path = os.path.dirname(os.path.realpath(__file__))
        self.par_tg = os.path.join(module_path, 'test_data/spanish-par-linux-3.2-utf8.bin')
        self.extractor_pos = POSFeatureExtractor( tagger=self.tagger, par_file_src=self.par_src, par_file_tg=self.par_tg )
        self.extractor_no_pos = POSFeatureExtractor()


    def test_pos_in_obj(self):
        obj = {'token':u'a', 'index':0, 'target':[u'a',u'boy',u'hits',u'a',u'dog'], 'source':[u'un', u'garcon',u'frappe', u'un', u'chien'], 'target_pos':['DT','NN','VBZ', 'DT', 'NN'], 'source_pos':['DT','NN','VBZ', 'DT', 'NN'], 'alignments':[[0],[1],[2],[3],[4]]}
        obj_no_align = {'token':u'a', 'index':0, 'target':[u'a',u'boy',u'hits',u'a',u'dog'], 'source':[u'un', u'garcon',u'frappe', u'un', u'chien'], 'target_pos':['DT','NN','VBZ', 'DT', 'NN'], 'source_pos':['DT','NN','VBZ', 'DT', 'NN']}
        (t1, s1) = self.extractor_no_pos.get_features(obj)
        (t2, s2) = self.extractor_no_pos.get_features(obj_no_align)
        self.assertEqual(t1, u'DT')
        self.assertEqual(s1, ['DT'])
        self.assertEqual(t2, 'DT')
        self.assertEqual(s2, [])

    def test_tag_on_the_fly(self):
        # tagging on the fly, adding tagging to the object
        obj = {'token':u'niño', 'index':1, 'source':[u'a',u'boy',u'hits',u'a',u'dog'], 'target':[ u'un', u'niño', u'vapulea', u'un', u'perro'], 'alignments':[[0],[1],[2],[3],[4]]}
        (t1, s1) = self.extractor_pos.get_features(obj)
        self.assertEqual(t1, 'NC')
        self.assertEqual(s1, [u'NN'])
        self.assertTrue(obj.has_key('target_pos'))
        self.assertTrue(obj.has_key('source_pos'))

    def test_no_tagger(self):
        # no information for tagging
        err = StringIO.StringIO()
        sys.stderr = err

        obj2 = {'token':u'niño', 'index':1, 'source':[u'a',u'boy',u'hits',u'a',u'dog'], 'target':[ u'un', u'niño', u'vapulea', u'un', u'perro'], 'alignments':[[0],[1],[2],[3],[4]]}
        (t2, s2) = self.extractor_no_pos.get_features(obj2)
        self.assertEqual( err.getvalue(), 'Tagging script and parameter file should be provided\nTagging script and parameter file should be provided\n' )
        err.close()
        self.assertEqual(t2, u'')
        self.assertEqual(s2, [])

    def test_only_target_tagging(self):
        # no alignments
        obj = {'token':u'niño', 'index':1, 'source':[u'a',u'boy',u'hits',u'a',u'dog'], 'target':[ u'un', u'niño', u'vapulea', u'un', u'perro']}
        (t1, s1) = self.extractor_pos.get_features(obj)
        self.assertEqual(t1, 'NC')
        self.assertEqual(s1, [])


if __name__ == '__main__':
    unittest.main()

