import unittest

from marmot.features.wordnet_feature_extractor import WordnetFeatureExtractor

class WordnetFeatureExtractorTests(unittest.TestCase):

    def setUp(self):
        self.wordnet_extractor = WordnetFeatureExtractor(src_lang='fre', tg_lang='en')
#        self.wordnet_extractor_fr = WordnetFeatureExtractor()

    def test_get_features(self):
        obj = {'token':u'hits', 'index':2, 'target':[u'a',u'boy',u'hits',u'a',u'dog'], 'source':[u'un', u'garcon',u'frappe', u'un', u'chien'], 'target_pos':['DT','NN','VBZ', 'DT', 'NN'], 'source_pos':['DT','NN','VBZ', 'DT', 'NN'], 'alignments':[[0],[1],[3],[2],[4]]}
        wn_src, wn_tg = self.wordnet_extractor.get_features(obj)
        self.assertEqual(wn_src, 9)
        self.assertEqual(wn_tg, 24)

    def test_no_source(self):
        obj = {'token':u'hits', 'index':2, 'target':[u'a',u'boy',u'hits',u'a',u'dog'], 'target_pos':['DT','NN','VBZ', 'DT', 'NN']}
        wn_src, wn_tg = self.wordnet_extractor.get_features(obj)
        self.assertEqual(wn_src, 0)
        self.assertEqual(wn_tg, 24)

    def test_no_alignment(self):
        obj = {'token':u'hits', 'index':2, 'target':[u'a',u'boy',u'hits',u'a',u'dog'], 'source':[u'un', u'garcon',u'frappe', u'un', u'chien'], 'target_pos':['DT','NN','VBZ', 'DT', 'NN'], 'source_pos':['DT','NN','VBZ', 'DT', 'NN']}
        wn_src, wn_tg = self.wordnet_extractor.get_features(obj)
        self.assertEqual(wn_src, 0)
        self.assertEqual(wn_tg, 24)

    def test_multi_alignment(self):
        obj = {'token':u'hits', 'index':2, 'target':[u'a',u'boy',u'hits',u'a',u'dog'], 'source':[u'un', u'garcon',u'frappe', u'un', u'chien'], 'target_pos':['DT','NN','VBZ', 'DT', 'NN'], 'source_pos':['DT','NN','VBZ', 'DT', 'NN'], 'alignments':[[0],[1],[3, 4],[2],[4]]}
        wn_src, wn_tg = self.wordnet_extractor.get_features(obj)
        self.assertEqual(wn_src, 9)
        self.assertEqual(wn_tg, 24)


#    def test_no_pos_fr(self):
if __name__ == '__main__':
    unittest.main()

