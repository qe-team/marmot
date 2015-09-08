#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
from marmot.features.phrase.phrase_alignment_feature_extractor import PhraseAlignmentFeatureExtractor


# test a class which extracts source and target token count features, and the source/target token count ratio
class PhraseAlignmentFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        self.extractor = PhraseAlignmentFeatureExtractor('/export/data/varvara/marmot/marmot/experiment/test_data/europarl_align_model')

    def test_get_features(self):
        obj = {'source': ['a', 'boy', 'hits', 'the', 'dog'],
               'target': ['uno', 'nino', 'abati', 'el', 'perro'],
               'alignments': [[], [0, 1], [2, 3], [3], [4]],
               'token': ['uno', 'nino', 'abati'],
               'index': (0, 3),
               'source_token': ['a', 'boy', 'hits'],
               'source_index': (0, 3)}

        (n_unaligned, n_multi, align_num) = self.extractor.get_features(obj)
        self.assertAlmostEqual(n_unaligned, 0.333333333)
        self.assertAlmostEqual(n_multi, 0.666666666)
        self.assertAlmostEqual(align_num, 1.33333333)


if __name__ == '__main__':
    unittest.main()
