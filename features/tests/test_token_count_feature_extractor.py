#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import os
from marmot.features.token_count_feature_extractor import TokenCountFeatureExtractor


# test a class which extracts source and target token count features, and the source/target token count ratio
class TokenCountFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(__file__)
        self.module_path = module_path
        self.tokenCounter = TokenCountFeatureExtractor()

    def test_get_features(self):
        # { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
        vector = self.tokenCounter.get_features( {'token':'a', 'index':2, 'source':[u'c',u'\'',u'est',u'un',u'garçon'], 'target':[u'this',u'is',u'a',u'boy',u'.'], 'tag':'G'})
        # the tokenCounter outputs three features
        self.assertEqual(len(vector), 3)
        self.assertEqual(vector[0], 5.0)
        self.assertEqual(vector[1], 5.0)
        self.assertEqual(vector[2], 1.0)


if __name__ == '__main__':
    unittest.main()