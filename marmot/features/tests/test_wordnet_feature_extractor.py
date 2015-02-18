import unittest

from marmot.features.wordnet_feature_extractor import WordnetFeatureExtractor

class WordnetFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        self.wordnet_extractor = WordnetFeatureExtractor()
        self.wordnet_extractor_fr = WordnetFeatureExtractor()
        pass

    def test_no_pos_eng(self):
        obj = {'token':u'hits', 'index':2, 'target':[u'a',u'boy',u'hits',u'a',u'dog'], 'source':[u'un', u'garcon',u'frappe', u'un', u'chien'], 'target_pos':['DT','NN','VBZ', 'DT', 'NN'], 'source_pos':['DT','NN','VBZ', 'DT', 'NN'], 'alignments':[[0],[1],[3],[2],[4]]}
        wn_feature = self.wordnet_extractor.get_features(obj)
        self.assertEqual(wn_feature, 24)

#    def test_no_pos_fr(self):
if __name__ == '__main__':
    unittest.main()

