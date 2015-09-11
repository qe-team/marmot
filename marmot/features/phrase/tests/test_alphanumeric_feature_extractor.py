#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import os
from marmot.features.phrase.alphanumeric_feature_extractor import AlphaNumericFeatureExtractor


# test a class which extracts source and target token count features, and the source/target token count ratio
class AlphaNumericFeatureExtractorTests(unittest.TestCase):
    
    def setUp(self):
        self.extractor = AlphaNumericFeatureExtractor()

    def test_get_features(self):
        obj = {'target':['a', 'boy', 'hits', '3', 'dogs', 'a11i', 'o8', 'www1'], 
               'source':['un', 'garcon', 'bate', '3', 'ou', '4', 'chiens', 'b2b'], 
               'token':['3', 'dogs', 'a11i', 'o8', 'www1'], 
               'index': (3, 8), 
               'source_token': ['3', 'ou', '4', 'chiens', 'b2b'], 
               'source_index':(3, 8)}

        (src_num, tg_num, num_ratio, src_alnum, tg_alnum, alnum_ratio) = self.extractor.get_features(obj)
        self.assertAlmostEqual(src_num, 0.4)
        self.assertAlmostEqual(tg_num, 0.2)
        self.assertAlmostEqual(num_ratio, 0.2)
        self.assertAlmostEqual(src_alnum, 0.2)
        self.assertAlmostEqual(tg_alnum, 0.6)
        self.assertAlmostEqual(alnum_ratio, 0.4)

    def test_get_features_no_src(self):
        obj_no_src = {'target':['a', 'boy', 'hits', '3', 'dogs', 'a11i', 'o8', 'www1'], 
               'source':['un', 'garcon', 'bate', '3', 'ou', '4', 'chiens', 'b2b'], 
               'token':['3', 'dogs', 'a11i', 'o8', 'www1'], 
               'index': (3, 8), 
               'source_token': [], 
               'source_index':[]}

        (src_num, tg_num, num_ratio, src_alnum, tg_alnum, alnum_ratio) = self.extractor.get_features(obj_no_src)
        self.assertAlmostEqual(src_num, 0)
        self.assertAlmostEqual(tg_num, 0.2)
        self.assertAlmostEqual(num_ratio, 0.2)
        self.assertAlmostEqual(src_alnum, 0)
        self.assertAlmostEqual(tg_alnum, 0.6)
        self.assertAlmostEqual(alnum_ratio, 0.6)



if __name__ == '__main__':
    unittest.main()
