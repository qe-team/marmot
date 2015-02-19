#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import os
from marmot.features.lm_feature_extractor import LMFeatureExtractor


# test a class which extracts source and target token count features, and the source/target token count ratio
class LMFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(os.path.realpath(__file__))
        self.module_path = module_path
        self.lm3Extractor = LMFeatureExtractor(os.path.join(module_path, 'test_data/training.txt'))
        self.lm5Extractor = LMFeatureExtractor(os.path.join(module_path, 'test_data/training.txt'), order=5)


    def test_get_features(self):
        # { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
        (left3, right3, back_l, back_m, back_r) = self.lm3Extractor.get_features({'token': 'for', 'index': 6, 'source': [u'c',u'\'',u'est',u'un',u'garçon'], 'target': [u'It', u'becomes', u'more', u'and', u'more', u'difficult', u'for', u'us', u'to', u'protect', u'her', u'brands', u'in', u'China', '.'], 'tag':'G'})
        (left5, right5, back_l, back_m, back_r) = self.lm5Extractor.get_features({'token':'for', 'index':6, 'source':[u'c',u'\'',u'est',u'un',u'garçon'], 'target':[u'It', u'becomes', u'more', u'and', u'more', u'difficult', u'for', u'us', u'to', u'protect', u'her', u'brands', u'in', u'China', '.'], 'tag':'G'})
        # TODO: this is not a test
        self.assertEqual(left3, 3)
        self.assertEqual(right3, 2)
        self.assertEqual(left5, 5)
        self.assertEqual(right5, 2)

    def test_backoff(self):
        context_obj = {'token':'for', 'index':2, 'source':[u'c',u'\'',u'est',u'un',u'garçon'], 'target':[u'It', u'becomes', u'more', u'more', u'difficult', u'for', u'us', u'to', u'protect', u'her', u'brands', u'in', u'China', '.'], 'tag':'G'}
        (left3, right3, back_l, back_m, back_r) = self.lm3Extractor.get_features(context_obj)
        self.assertAlmostEqual(back_l, 1.0)
        self.assertAlmostEqual(back_m, 0.4)
        self.assertAlmostEqual(back_r, 0.6)
        context_obj = {'token':'telescope', 'index':6, 'source':[u'c',u'\'',u'est',u'un',u'garçon'], 'target':[u'One', u'of', u'the', u'tasks', u'to', u'the', u'telescope', u'China', u'GAGARIN'], 'tag':'G'}
        (left3, right3, back_l, back_m, back_r) = self.lm3Extractor.get_features(context_obj)
        self.assertAlmostEqual(back_l, 0.8)
        self.assertAlmostEqual(back_m, 0.4)
        self.assertAlmostEqual(back_r, 0.1)
        context_obj = {'token':'UUUU', 'index':2, 'source':[u'c',u'\'',u'est',u'un',u'garçon'], 'target':[u'OOOOO', u'AAAAA', u'UUUU', u'China', u'telescope'], 'tag':'G'}
        (left3, right3, back_l, back_m, back_r) = self.lm3Extractor.get_features(context_obj)
        self.assertAlmostEqual(back_l, 0.1)
        self.assertAlmostEqual(back_m, 0.2)
        self.assertAlmostEqual(back_r, 0.3)


if __name__ == '__main__':
    unittest.main()
