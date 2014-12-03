#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import os
from marmot.features.special_list_feature_extractor import SpecListFeatureExtractor


# test a class which extracts source and target token count features, and the source/target token count ratio
class TokenCountFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(os.path.realpath(__file__))
        self.module_path = module_path
        self.spec_list = SpecListFeatureExtractor(language='english')
        self.custom_list = SpecListFeatureExtractor(punctuation=',.:;()', stopwords=['Sam'])


    def test_get_features(self):
        # { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
        #( is stopword, is punctuation, is proper name, is digit )
        (s, p, pr, d) = self.spec_list.get_features( {'token':'a', 'index':2, 'source':[u'c',u'\'',u'est',u'un',u'garçon'], 'target':[u'this',u'is',u'a',u'boy',u'.'], 'tag':'G'} )
        self.assertEqual(s, 1)
        self.assertEqual(p, 0)
        self.assertEqual(pr, 0)
        self.assertEqual(d, 0)
        (s, p, pr, d) = self.custom_list.get_features( {'token':'Sam', 'index':2, 'source':[u'c',u'\'',u'est',u'un',u'garçon'], 'target':[u'this',u'is',u'Sam',u'!'], 'tag':'G'} )
        self.assertEqual(pr, 1)
        self.assertEqual(s, 1)
        (s, p, pr, d) = self.custom_list.get_features( {'token':'!', 'index':3, 'source':[u'c',u'\'',u'est',u'un',u'garçon'], 'target':[u'this',u'is',u'Sam',u'!'], 'tag':'G'} )
        self.assertEqual(p, 0)
        (s, p, pr, d) = self.spec_list.get_features( {'token':'33', 'index':2, 'source':[u'c',u'\'',u'est',u'un',u'garçon'], 'target':[u'this',u'is',u'33',u'!'], 'tag':'G'} )
        self.assertEqual(d, 1)
 


if __name__ == '__main__':
    unittest.main()
