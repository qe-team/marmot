#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
from marmot.features.phrase.punctuation_feature_extractor import PunctuationFeatureExtractor


# test a class which extracts source and target token count features, and the source/target token count ratio
class PunctuationFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        self.extractor = PunctuationFeatureExtractor()

    def test_get_features(self):
        obj = {'target': ['a', ',', 'boy', ',', 'hits', '!', '3', 'dogs', ':', 'www1', '.', 'some', 'words', 'not', 'in', 'phrase'],
               'source': ['un', ',', 'garcon', ';', 'bate', '!', '!', '3', 'ou', '4', '?', 'chiens', '.', 'quelques', 'mots', 'inutils', 'lalala'],
               'token': [',', 'boy', ',', 'hits', '!', '3', 'dogs', ':', ':', '.'],
               'index': (1, 11),
               'source_token': [',', 'garcon', ';', 'bate', '!', '!', '3', 'ou', '4', '?', 'chiens', '.'],
               'source_index': (1, 13)}

        all_features = self.extractor.get_features(obj)
        self.assertEqual(all_features[0], 0)
        self.assertEqual(all_features[1], -1)
        self.assertEqual(all_features[2], -2)
        self.assertEqual(all_features[3], 1)
        self.assertEqual(all_features[4], 1)
        self.assertEqual(all_features[5], 1)
        self.assertAlmostEqual(all_features[6], 0.0)
        self.assertAlmostEqual(all_features[7], -0.1)
        self.assertAlmostEqual(all_features[8], -0.2)
        self.assertAlmostEqual(all_features[9], 0.1)
        self.assertAlmostEqual(all_features[10], 0.1)
        self.assertAlmostEqual(all_features[11], 0.1)
        self.assertAlmostEqual(all_features[12], 0.5)
        self.assertAlmostEqual(all_features[13], 0.6)
        self.assertAlmostEqual(all_features[14], 0.0)

        '''
        0 - 'diff_periods',
        1 - 'diff_commas',
        2 - 'diffe_colons',
        3 - 'diff_semicolons',
        4 - 'diff_questions',
        5 - 'diff_exclamations',
        6 - 'diff_periods_weighted',
        7 - 'diff_commas_weighted',
        8 - 'diffe_colons_weighted',
        9 - 'diff_semicolons_weighted',
        10 - 'diff_questions_weighted',
        11 - 'diff_exclamations_weighted',
        12 - 'percentage_punct_source',
        13 - 'percentage_punct_target',
        14 - 'diff_punct'
        '''
        obj_no_src = {'target': ['a', 'boy', 'hits', '3', 'dogs', 'a11i', 'o8', 'www1'],
                      'source': ['un', 'garcon', 'bate', '3', 'ou', '4', 'chiens', 'b2b'],
                      'token': ['3', 'dogs', 'a11i', 'o8', 'www1'],
                      'index': (3, 8),
                      'source_token': [],
                      'source_index': []}
        all_features = self.extractor.get_features(obj_no_src)
        self.assertEqual(all_features[0], 0)
        self.assertEqual(all_features[1], 0)
        self.assertEqual(all_features[2], 0)
        self.assertEqual(all_features[3], 0)
        self.assertEqual(all_features[4], 0)
        self.assertEqual(all_features[5], 0)
        self.assertAlmostEqual(all_features[6], 0.0)
        self.assertAlmostEqual(all_features[7], 0.0)
        self.assertAlmostEqual(all_features[8], 0.0)
        self.assertAlmostEqual(all_features[9], 0.0)
        self.assertAlmostEqual(all_features[10], 0.0)
        self.assertAlmostEqual(all_features[11], 0.0)
        self.assertAlmostEqual(all_features[12], 0.0)
        self.assertAlmostEqual(all_features[13], 0.0)
        self.assertAlmostEqual(all_features[14], 0.0)


if __name__ == '__main__':
    unittest.main()
