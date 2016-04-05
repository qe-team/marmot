#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
from marmot.features.phrase.ne_feature_extractor import NEFeatureExtractor


# test a class which extracts source and target token count features, and the source/target token count ratio
class AlphaNumericFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        self.extractor = NEFeatureExtractor()

    def test_get_features(self):
        obj = {'target': ['a', 'boy', 'hits', '3', 'dogs', 'a11i', 'o8', 'www1'],
               'source': ['un', 'garcon', 'bate', '3', 'ou', '4', 'chiens', 'b2b'],
               'token': ['3', 'dogs', 'Tinky', 'and', 'Winky'],
               'index': (3, 8),
               'source_token': ['3', 'ou', 'Tinky', 'et', 'Winky'],
               'source_index': (3, 8)}

        (src_ne, tg_ne) = self.extractor.get_features(obj)
        self.assertEqual(src_ne, 1)
        self.assertEqual(tg_ne, 1)

    def test_get_features_no_src(self):
        obj_no_src = {'target': ['a', 'boy', 'hits', '3', 'dogs', 'a11i', 'o8', 'www1'],
               'source': ['un', 'garcon', 'bate', '3', 'ou', '4', 'chiens', 'b2b'],
               'token': ['3', 'dogs', 'Tinky', 'and', 'Winky'],
               'index': (3, 8),
               'source_token': [],
               'source_index': []}

        (src_ne, tg_ne) = self.extractor.get_features(obj_no_src)
        self.assertEqual(src_ne, 0)
        self.assertEqual(tg_ne, 1)

    def test_get_features_no_NE(self):
        obj_no_src = {'target': ['a', 'boy', 'hits', '3', 'dogs', 'a11i', 'o8', 'www1'],
               'source': ['un', 'garcon', 'bate', '3', 'ou', '4', 'chiens', 'b2b'],
               'token': ['3', 'dogs', 'aDa', 'and', 'ooO'],
               'index': (3, 8),
               'source_token': [],
               'source_index': []}

        (src_ne, tg_ne) = self.extractor.get_features(obj_no_src)
        self.assertEqual(src_ne, 0)
        self.assertEqual(tg_ne, 0)

if __name__ == '__main__':
    unittest.main()
