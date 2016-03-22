#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
from marmot.features.phrase.context_feature_extractor import ContextFeatureExtractor


# test a class which extracts source and target token count features, and the source/target token count ratio
class ContextFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        self.extractor = ContextFeatureExtractor()

    def test_get_features(self):
        obj = {'source': ['Edward', 'is', 'a', 'friend', 'of', 'mine'], 'target': ['Edward', 'est', 'mon', 'ami'], 'index': (0, 2), 'source_index': (0, 4), 'token': ['Edward', 'est'], 'source_token': ['Edward', 'is', 'a', 'friend']}
        contexts = self.extractor.get_features(obj)
        self.assertEqual(contexts[0], "<s>")
        self.assertEqual(contexts[1], "of")
        self.assertEqual(contexts[2], "<s>")
        self.assertEqual(contexts[3], "mon")


if __name__ == '__main__':
    unittest.main()
