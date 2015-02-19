import unittest
import yaml
import os

from marmot.features.alignment_feature_extractor import AlignmentFeatureExtractor
from marmot.features.pos_feature_extractor import POSFeatureExtractor
from marmot.features.google_translate_feature_extractor import GoogleTranslateFeatureExtractor
from marmot.features.source_lm_feature_extractor import SourceLMFeatureExtractor

from marmot.exceptions.no_data_error import NoDataError
from marmot.exceptions.no_resource_error import NoResourceError


class FeatureExtractorErrorTest(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(__file__)
        self.module_path = module_path
#        test_config = os.path.join(module_path, 'test_data/test_config.yaml')
#        
#        with open(test_config, "r") as cfg_file:
#            self.config = yaml.load(cfg_file.read())

    def test_alignment_no_source(self):
        alignmentFE = AlignmentFeatureExtractor()
        obj = {'token':u'hits', 'index':2, 'target':[u'a',u'boy',u'hits',u'a',u'dog']}
        with self.assertRaises(NoDataError):
            alignmentFE.get_features(obj)


    def test_alignment_no_target(self):
        alignmentFE = AlignmentFeatureExtractor()
        obj = {'token':u'hits', 'index':2, 'source':[u'un', u'garcon',u'frappe', u'un', u'chien']}
        with self.assertRaises(NoDataError):
            alignmentFE.get_features(obj)

    def test_alignment_no_alignments(self):
        alignmentFE = AlignmentFeatureExtractor()
        obj = {'token':u'hits', 'index':2, 'target':[u'a',u'boy',u'hits',u'a',u'dog'], 'source':[u'un', u'garcon',u'frappe', u'un', u'chien']}
        with self.assertRaises(NoDataError):
            alignmentFE.get_features(obj)

    def test_pos_no_source(self):
        posFE = POSFeatureExtractor(tagger=os.path.join(self.module_path, '../../experiment/tiny_test/tree-tagger'), par_file_src=os.path.join(self.module_path, '../../experiment/tiny_test/spanish-par-linux-3.2-utf8.bin'), par_file_tg=os.path.join(self.module_path, '../../experiment/tiny_test/english-utf8.par'))
        obj = {'token':u'hits', 'index':2, 'target':[u'a',u'boy',u'hits',u'a',u'dog']}
        with self.assertRaises(NoDataError):
            posFE.get_features(obj)
        
    def test_pos_no_target(self):
        posFE = POSFeatureExtractor()
        obj = {'token':u'hits', 'index':2, 'source':[u'un', u'garcon',u'frappe', u'un', u'chien']}
        with self.assertRaises(NoDataError):
            posFE.get_features(obj)

    def test_pos_no_tagger(self):
        posFE = POSFeatureExtractor()
        obj = {'token':u'hits', 'index':2, 'source':[u'un', u'garcon',u'frappe', u'un', u'chien'], 'target':[u'a',u'boy',u'hits',u'a',u'dog']}
        with self.assertRaises(NoResourceError):
            posFE.get_features(obj)

    def test_pos_no_tagger_params(self):
        posFE = POSFeatureExtractor(tagger='../../experiment/tiny_test/tree-tagger')
        obj = {'token':u'hits', 'index':2, 'source':[u'un', u'garcon',u'frappe', u'un', u'chien'], 'target':[u'a',u'boy',u'hits',u'a',u'dog']}
        with self.assertRaises(NoResourceError):
            posFE.get_features(obj)


    def test_google_no_source(self):
        gtFE = GoogleTranslateFeatureExtractor()
        obj = {'token':u'hits', 'index':2, 'target':[u'a',u'boy',u'hits',u'a',u'dog']}
        with self.assertRaises(NoDataError):
            gtFE.get_features(obj)

    def test_source_lm_no_source(self):
        slmFE = SourceLMFeatureExtractor(os.path.join(self.module_path, '../../experiment/tiny_test/europarl.1000.en'))
        obj = {'token':u'hits', 'index':2, 'target':[u'a',u'boy',u'hits',u'a',u'dog']}
        with self.assertRaises(NoDataError):
            slmFE.get_features(obj)

    def test_source_lm_no_alignments(self):
        slmFE = SourceLMFeatureExtractor(os.path.join(self.module_path, '../../experiment/tiny_test/europarl.1000.en'))
        obj = {'token':u'hits', 'index':2, 'target':[u'a',u'boy',u'hits',u'a',u'dog'], 'source':[u'un', u'garcon',u'frappe', u'un', u'chien']}
        with self.assertRaises(NoDataError):
            slmFE.get_features(obj)


if __name__ == '__main__':
        unittest.main()
