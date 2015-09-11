#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import os
from marmot.features.phrase.token_count_feature_extractor import TokenCountFeatureExtractor


# test a class which extracts source and target token count features, and the source/target token count ratio
class TokenCountFeatureExtractorTests(unittest.TestCase):
    
    def setUp(self):
        self.extractor = TokenCountFeatureExtractor()

    def test_get_features(self):
        obj = {'target':['a', 'boy', 'hits', 'a', 'dog'], 
               'source':['un', 'garcon', 'bate', 'un', 'chien'], 
               'token':['a', 'boy'], 
               'index': (0, 2), 
               'source_token': ['un', 'garcon'], 
               'source_index':(0, 2)}
        (tg_ph_len, src_ph_len, src_tg_ratio, tg_src_ratio, tg_token_len, src_token_len, token_occ) = self.extractor.get_features(obj)
        self.assertEqual(tg_ph_len, 2)
        self.assertEqual(src_ph_len, 2)
        self.assertEqual(src_tg_ratio, 1)
        self.assertEqual(tg_src_ratio, 1)
        self.assertEqual(tg_token_len, 2)
        self.assertEqual(src_token_len, 4)
        self.assertAlmostEqual(token_occ, 1.5)

    def test_get_features_no_src(self):
        obj = {'target':['a', 'boy', 'hits', 'a', 'dog'], 
               'source':['un', 'garcon', 'bate', 'un', 'chien'], 
               'token':['a', 'boy'], 
               'index': (0, 2), 
               'source_token': [], 
               'source_index':()}
        (tg_ph_len, src_ph_len, src_tg_ratio, tg_src_ratio, tg_token_len, src_token_len, token_occ) = self.extractor.get_features(obj)
        self.assertEqual(tg_ph_len, 2)
        self.assertEqual(src_ph_len, 0)
        self.assertEqual(src_tg_ratio, 0)
        self.assertEqual(tg_src_ratio, 0)
        self.assertEqual(tg_token_len, 2)
        self.assertEqual(src_token_len, 0)
        self.assertAlmostEqual(token_occ, 1.5)


if __name__ == '__main__':
    unittest.main()
