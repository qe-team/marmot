#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, glob
from subprocess import call
import unittest

from marmot.features.alignment_feature_extractor import AlignmentFeatureExtractor

class AlignmentFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        self.module_path = os.path.dirname(os.path.realpath(__file__))
        self.src_name = os.path.join(self.module_path,'../../preprocessing/tests/test_data/corpus.de.1000')
        self.tg_name = os.path.join(self.module_path,'../../preprocessing/tests/test_data/corpus.en.1000')
        self.aligner_no_model = AlignmentFeatureExtractor()
        self.aligner_no_model_2 = AlignmentFeatureExtractor(context_size=2)


    def test_alignment_in_obj(self):
        obj = {'token':u'hits', 'index':2, 'target':[u'a',u'boy',u'hits',u'a',u'dog'], 'source':[u'un', u'garcon',u'frappe', u'un', u'chien'], 'target_pos':['DT','NN','VBZ', 'DT', 'NN'], 'source_pos':['DT','NN','VBZ', 'DT', 'NN'], 'alignments':[[0],[1],[3],[2],[4]]}
        (cont_word, left, right) = self.aligner_no_model.get_features(obj)
        self.assertEqual(cont_word, u'un')
        self.assertEqual(left, u'frappe')
        self.assertEqual(right, u'chien')
        (cont_word, left, right) = self.aligner_no_model_2.get_features(obj)
        self.assertEqual(left, u'garcon frappe')
        self.assertEqual(right, u'chien _END_')


    def test_alignment_on_the_fly(self):
        obj = {'token':u'boy', 'index':1, 'source':[u'ein', u'junge', u'schlägt', u'einen', u'Hund'], 'target':[u'a', u'boy', u'hits', u'a', u'dog']}
        aligner_corpus = AlignmentFeatureExtractor(src_file = self.src_name, tg_file = self.tg_name)
        (cont_word, left, right) = aligner_corpus.get_features(obj)
        self.assertTrue(obj.has_key('alignments'))
        self.assertEqual(cont_word, u'junge')
        for a_file in glob.glob('align_model.*'):
            call(['rm', a_file])
        for a_file in glob.glob(os.path.basename(self.src_name)+'_'+os.path.basename(self.tg_name)+'*'):
            call(['rm', a_file])


    def test_align_model_in_extractor(self):
        obj = {'token':u'boy', 'index':1, 'source':[u'ein', u'junge', u'schlägt', u'einen', u'Hund'], 'target':[u'a', u'boy', u'hits', u'a', u'dog']}
        aligner_model = AlignmentFeatureExtractor( align_model = os.path.join(self.module_path, 'test_data/alignments/align_model') )
        (cont_word, left, right) = aligner_model.get_features(obj)
        self.assertTrue(obj.has_key('alignments'))
        self.assertEqual(cont_word, u'junge')


    def test_unaligned(self):
        obj = {'token':u'hits', 'index':2, 'target':[u'a',u'boy',u'hits',u'a',u'dog'], 'source':[u'un', u'garcon',u'frappe', u'un', u'chien'], 'target_pos':['DT','NN','VBZ', 'DT', 'NN'], 'source_pos':['DT','NN','VBZ', 'DT', 'NN'], 'alignments':[[0],[1],[],[2],[4]]}
        (cont_word, left, right) = self.aligner_no_model.get_features(obj)
        self.assertEqual(cont_word, u'Unaligned')
        self.assertEqual(left, u'Unaligned')
        self.assertEqual(right, u'Unaligned')



if __name__ == '__main__':
    unittest.main()

