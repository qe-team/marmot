#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import os
from marmot.features.phrase.ngram_frequencies_feature_extractor import NgramFrequenciesFeatureExtractor


# test a class which extracts source and target token count features, and the source/target token count ratio
class NgramFrequenciesFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        self.extractor = NgramFrequenciesFeatureExtractor(os.getcwd(), ngram_count_file='test_data/ngram_counts')

    def test_extractor_creation(self):
        extractor = NgramFrequenciesFeatureExtractor(os.getcwd(), corpus='test_data/corpus.en')
        self.assertEqual(extractor.ngrams[1]['must'], 783)
        self.assertEqual(extractor.ngrams[2]['drop of'], 2)
        self.assertEqual(extractor.ngrams[3]['is something that'], 11)

    def test_get_features(self):
        context = {'token': ['eso', 'es', 'naturally', 'unacceptable', 'ggg'], 'source_token': ['naturally', 'unacceptable', 'thing', 'is']}
        features = self.extractor.get_features(context)
        self.assertAlmostEqual(features[-3], 1.0)
        self.assertAlmostEqual(features[-2], 0.5)
        self.assertAlmostEqual(features[-1], 0.0)


if __name__ == '__main__':
        unittest.main()
