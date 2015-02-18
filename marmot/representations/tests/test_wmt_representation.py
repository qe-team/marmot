#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import yaml
import os
import shutil

from marmot.representations.wmt_representation_generator import WMTRepresentationGenerator
from marmot.experiment.import_utils import build_object


class WMTRepresentationGeneratorTests(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(__file__)
        self.module_path = module_path
        test_config = os.path.join(module_path, 'test_config.yaml')

        with open(test_config, "r") as cfg_file:
            self.config = yaml.load(cfg_file.read())

    def test_load_from_config(self):
        generator = build_object(self.config['representations']['training'][0])
        data_obj = generator.generate({})
        self.assertTrue('target' in data_obj)
        self.assertTrue('source' in data_obj)
        self.assertTrue('tags' in data_obj)
        self.assertTrue(len(data_obj['target']) == len(data_obj['source']))
        self.assertTrue(len(data_obj['target']) == len(data_obj['tags']))

    def test_no_saved_files(self):
        generator = WMTRepresentationGenerator('../../experiment/tiny_test/EN_ES.tgt_ann.train', '../../experiment/tiny_test/EN_ES.source.train')
        data_obj = generator.generate({})
        self.assertTrue('target' in data_obj)
        self.assertTrue('source' in data_obj)
        self.assertTrue('tags' in data_obj)
        self.assertTrue(len(data_obj['target']) == len(data_obj['source']))
        self.assertTrue(len(data_obj['target']) == len(data_obj['tags']))

    def test_save_files(self):
        tmp = os.path.join(self.module_path, 'tmp_dir')
        generator = WMTRepresentationGenerator(os.path.join(self.module_path, '../../experiment/tiny_test/EN_ES.tgt_ann.train'), os.path.join(self.module_path, '../../experiment/tiny_test/EN_ES.source.train'), tmp_dir=tmp, persist=True)
        data_obj = generator.generate({})
        target = os.path.join(tmp, 'EN_ES.tgt_ann.train.target')
        tags = os.path.join(tmp, 'EN_ES.tgt_ann.train.tags')
        source = os.path.join(tmp, 'EN_ES.source.train.txt')
        self.assertTrue(os.path.exists(tmp) and os.path.isdir(tmp))
        self.assertTrue(os.path.exists(target) and os.path.isfile(target))
        self.assertTrue(os.path.exists(tags) and os.path.isfile(tags))
        self.assertTrue(os.path.exists(source) and os.path.isfile(source))
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

