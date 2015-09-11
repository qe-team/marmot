#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
from marmot.features.phrase.pos_feature_extractor import POSFeatureExtractor


# test a class which extracts source and target token count features, and the source/target token count ratio
class POSFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        self.extractor = POSFeatureExtractor('english', 'spanish')

    def test_get_features(self):
        obj = {'source': ['a', 'boy', 'hits', 'the', 'small', 'dog', 'severely'],
               'target': ['uno', 'nino', 'abati', 'el', 'perro'],
               'alignments': [[], [0, 1], [2, 3], [3], [4]],
               'target_pos': ['ART', 'NC', 'VLfin', 'ART', 'NC'],
               'source_pos': ['DT', 'NN', 'VBZ', 'DT', 'JJ', 'NN', 'RB'],
               'token': ['uno', 'perro'],
               'index': (3, 5),
               'source_token': ['the', 'small', 'dog', 'severely'],
               'source_index': (3, 7)}

        '''
        0 - 'percentage_content_words_src',
        1 - 'percentage_content_words_tg',
        2 - 'percentage_verbs_src',
        3 - 'percentage_verbs_tg',
        4 - 'percentage_nouns_src',
        5 - 'percentage_nouns_tg',
        6 - 'percentage_pronouns_src',
        7 - 'percentage_pronouns_tg',
        8 - 'ratio_content_words_src_tg',
        9 - 'ratio_verbs_src_tg',
        10 - 'ratio_nouns_src_tg',
        11 - 'ratio_pronouns_src_tg'
        '''
        all_pos = self.extractor.get_features(obj)
        self.assertAlmostEqual(all_pos[0], 0.75)
        self.assertAlmostEqual(all_pos[1], 0.5)
        self.assertAlmostEqual(all_pos[2], 0.0)
        self.assertAlmostEqual(all_pos[3], 0.0)
        self.assertAlmostEqual(all_pos[4], 0.25)
        self.assertAlmostEqual(all_pos[5], 0.5)
        self.assertAlmostEqual(all_pos[6], 0.0)
        self.assertAlmostEqual(all_pos[7], 0.0)
        self.assertAlmostEqual(all_pos[8], 1.5)
        self.assertAlmostEqual(all_pos[9], 1.0)
        self.assertAlmostEqual(all_pos[10], 0.5)
        self.assertAlmostEqual(all_pos[11], 1.0)


if __name__ == '__main__':
    unittest.main()
