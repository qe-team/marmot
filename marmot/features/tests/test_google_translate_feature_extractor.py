#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import os
from marmot.features.google_translate_feature_extractor import GoogleTranslateFeatureExtractor


class GoogleTranslateFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        self.gs_extractor_en = GoogleTranslateFeatureExtractor(lang='en')
        self.gs_extractor_fr = GoogleTranslateFeatureExtractor(lang='fr')

    def test_get_features(self):
        gt1 = self.gs_extractor_en.get_features({'token':'short', 'index':3, 'source':[u'c',u'\'',u'est',u'une',u'courte', u'phrase'], 'target':[u'this', u'is', u'a', u'short', u'sentence', u'.'], 'tag':'G'})
        self.assertEqual([1], gt1)
        gt2 = self.gs_extractor_en.get_features({'token':'little', 'index':3, 'source':[u'c',u'\'',u'est',u'une',u'courte', u'phrase'], 'target':[u'this', u'is', u'a', u'little', u'sentence', u'.'], 'tag':'G'})
        self.assertEqual([0], gt2)

    def test_no_source(self):
        gt = self.gs_extractor_en.get_features({'token':'short', 'index':3, 'target':[u'this', u'is', u'a', u'short', u'sentence', u'.'], 'tag':'G'})
        self.assertEqual([], gt)


if __name__ == '__main__':
    unittest.main()
