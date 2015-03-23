#!/usr/bin/python
# -*- coding: utf-8 -*-
import unittest

from marmot.features.target_token_feature_extractor import TargetTokenFeatureExtractor


class AlignmentFeatureExtractorTests(unittest.TestCase):

    def test_get_features(self):
        obj = {'token': u'hits', 'index': 2, 'target': [u'a',u'boy',u'hits',u'a',u'dog'], 'source': [u'un', u'garcon',u'frappe', u'un', u'chien'], 'target_pos': ['DT','NN','VBZ', 'DT', 'NN'], 'source_pos': ['DT','NN','VBZ', 'DT', 'NN'], 'alignments': [[0],[1],[3],[2],[4]]}
        extractor = TargetTokenFeatureExtractor()
        [token, left, right] = extractor.get_features(obj)
        self.assertEqual(token, u'hits')
        self.assertEqual(left, u'boy')
        self.assertEqual(right, u'a')

    def test_get_features_two_words(self):
        obj = {'token': u'hits', 'index': 2, 'target': [u'a',u'boy',u'hits',u'a',u'dog'], 'source': [u'un', u'garcon',u'frappe', u'un', u'chien'], 'target_pos': ['DT','NN','VBZ', 'DT', 'NN'], 'source_pos': ['DT','NN','VBZ', 'DT', 'NN'], 'alignments': [[0],[1],[3],[2],[4]]}
        extractor = TargetTokenFeatureExtractor(context_size=2)
        [token, left, right] = extractor.get_features(obj)
        self.assertEqual(token, u'hits')
        self.assertEqual(left, u'a boy')
        self.assertEqual(right, u'a dog')

    def test_first_el(self):
        obj = {'token': u'a', 'index': 0, 'target': [u'a',u'boy',u'hits',u'a',u'dog'], 'source': [u'un', u'garcon',u'frappe', u'un', u'chien'], 'target_pos': ['DT','NN','VBZ', 'DT', 'NN'], 'source_pos': ['DT','NN','VBZ', 'DT', 'NN'], 'alignments': [[0],[1],[3],[2],[4]]}
        extractor = TargetTokenFeatureExtractor(context_size=2)
        [token, left, right] = extractor.get_features(obj)
        self.assertEqual(token, u'a')
        self.assertEqual(left, u'_START_ _START_')
        self.assertEqual(right, u'boy hits')


if __name__ == '__main__':
    unittest.main()
