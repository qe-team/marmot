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
        (left3, right3) = self.lm3Extractor.get_features( {'token':'for', 'index':6, 'source':[u'c',u'\'',u'est',u'un',u'garçon'], 'target':[u'It', u'becomes', u'more', u'and', u'more', u'difficult', u'for', u'us', u'to', u'protect', u'her', u'brands', u'in', u'China', '.'], 'tag':'G'})
        (left5, right5) = self.lm3Extractor.get_features( {'token':'for', 'index':6, 'source':[u'c',u'\'',u'est',u'un',u'garçon'], 'target':[u'It', u'becomes', u'more', u'and', u'more', u'difficult', u'for', u'us', u'to', u'protect', u'her', u'brands', u'in', u'China', '.'], 'tag':'G'})
        # TODO: this is not a test
        self.assertTrue(left3, 3)
        self.assertTrue(right3, 2)
        self.assertTrue(left5, 5)
        self.assertTrue(right5, 2)


if __name__ == '__main__':
    unittest.main()
