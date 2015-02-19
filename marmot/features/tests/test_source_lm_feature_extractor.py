#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import os
from marmot.features.source_lm_feature_extractor import SourceLMFeatureExtractor


class LMFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(os.path.realpath(__file__))
        self.module_path = module_path
        self.lm3Extractor = SourceLMFeatureExtractor(os.path.join(module_path, 'test_data/training.txt'))
        self.lm5Extractor = SourceLMFeatureExtractor(os.path.join(module_path, 'test_data/training.txt'), order=5)

    def test_get_features(self):
        # { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
        (left3, right3) = self.lm3Extractor.get_features({'token': 'est', 'index': 2, 'target': [u'c',u'\'',u'est',u'un',u'garçon'], 'source': [u'It', u'becomes', u'more', u'and', u'more', u'difficult', u'for', u'us', u'to', u'protect', u'her', u'brands', u'in', u'China', '.'], 'tag':'G', 'alignments': [[], [], [6], [], []]})
        (left5, right5) = self.lm5Extractor.get_features({'token':'est', 'index':2, 'target':[u'c',u'\'',u'est',u'un',u'garçon'], 'source':[u'It', u'becomes', u'more', u'and', u'more', u'difficult', u'for', u'us', u'to', u'protect', u'her', u'brands', u'in', u'China', '.'], 'tag':'G', 'alignments': [[], [], [6], [], []]})
        self.assertEqual(left3, 3)
        self.assertEqual(right3, 2)
        self.assertEqual(left5, 5)
        self.assertEqual(right5, 2)

    # TODO: if source or alignment don't exist, an error should be thrown 
    def test_no_source(self):
        features = self.lm3Extractor.get_features({'token': 'est', 'index': 2, 'target': [u'c',u'\'',u'est',u'un',u'garçon'], 'tag':'G'})
        self.assertEqual(features, [])

    def test_no_alignments(self):
        features = self.lm3Extractor.get_features({'token': 'est', 'index': 2, 'target': [u'c',u'\'',u'est',u'un',u'garçon'], 'source': [u'It', u'becomes', u'more', u'and', u'more', u'difficult', u'for', u'us', u'to', u'protect', u'her', u'brands', u'in', u'China', '.'], 'tag':'G'})
        self.assertEqual(features, [])

    def test_unaligned(self):
        left_ngram, right_ngram = self.lm3Extractor.get_features({'token': 'est', 'index': 2, 'target': [u'c',u'\'',u'est',u'un',u'garçon'], 'source': [u'this', u'is', u'a', u'boy'], 'alignments':[[0], [1], [], [3], [4]], 'tag':'G'})
        self.assertEqual(left_ngram, 0)
        self.assertEqual(right_ngram, 0)

    def test_multi_alignment(self):
        (left3, right3) = self.lm3Extractor.get_features({'token': 'est', 'index': 2, 'target': [u'c',u'\'',u'est',u'un',u'garçon'], 'source': [u'It', u'becomes', u'more', u'and', u'more', u'difficult', u'for', u'us', u'to', u'protect', u'her', u'brands', u'in', u'China', '.'], 'tag':'G', 'alignments': [[], [], [6, 7], [], []]})
        self.assertEqual(left3, 2)
        self.assertEqual(right3, 2)


if __name__ == '__main__':
    unittest.main()
