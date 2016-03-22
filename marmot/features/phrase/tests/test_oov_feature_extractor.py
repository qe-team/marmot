#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
from marmot.features.phrase.oov_feature_extractor import OOVFeatureExtractor


# test a class which extracts source and target token count features, and the source/target token count ratio
class OOVFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        self.extractor = OOVFeatureExtractor('test_data/corpus.en')

    def test_oov(self):
        obj = {'source': ['Edward', 'is', 'a', 'friend', 'of', 'mine'], 'target': ['Edward', 'est', 'mon', 'ami'], 'index': (0, 4), 'source_index': (0, 2), 'token': ['Edward', 'est'], 'source_token': ['Edward', 'is', 'a', 'friend']}
        self.assertEqual(self.extractor.get_features(obj)[0], 1)
        obj2 = {'source': ['he', 'is', 'a', 'friend', 'of', 'mine'], 'target': ['Il', 'est', 'mon', 'ami'], 'index': (0, 4), 'source_index': (0, 2), 'token': ['Il', 'est'], 'source_token': ['he', 'is', 'a', 'friend']}
        self.assertEqual(self.extractor.get_features(obj2)[0], 0)

if __name__ == '__main__':
    unittest.main()
