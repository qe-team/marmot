#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import os
from marmot.features.phrase.num_translations_feature_extractor import NumTranslationsFeatureExtractor


# test a class which extracts source and target token count features, and the source/target token count ratio
class NumTranslationsFeatureExtractorTests(unittest.TestCase):
    
    def setUp(self):
        self.extractor = NumTranslationsFeatureExtractor('/export/data/varvara/europarl-sys/english_spanish/tiny/model/lex.1.f2e', '/export/data/varvara/europarl-sys/english_spanish/tiny/corpus/nc.clean.1.en')

    def test_get_features(self):
        obj = {'source':['a', 'boy', 'hits', 'the', 'dog'], 
               'target':['uno', 'nino', 'abati', 'el', 'perro'], 
               'token':['el'], 
               'index': (3, 4), 
               'source_token': ['the'],
               'source_index':(3, 4)}

        (f_001, f_005, f_01, f_02, f_05, f_001_w, f_005_w, f_01_w, f_02_w, f_05_w) = self.extractor.get_features(obj)
        self.assertEqual(f_001, 7)
        self.assertEqual(f_005, 6)
        self.assertEqual(f_01, 3)
        self.assertEqual(f_02, 2)
        self.assertEqual(f_05, 0)
        self.assertAlmostEqual(f_001_w, 0.52421775)
        self.assertAlmostEqual(f_005_w, 0.4493295)
        self.assertAlmostEqual(f_01_w, 0.22466475)
        self.assertAlmostEqual(f_02_w, 0.1497765)
        self.assertAlmostEqual(f_05_w, 0.0)


if __name__ == '__main__':
    unittest.main()
