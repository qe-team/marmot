#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import yaml
import os
import shutil

import marmot
from marmot.representations.wmt_representation_generator import WMTRepresentationGenerator
from marmot.experiment.import_utils import build_object

def join_with_module_path(loader, node):
    """ define custom tag handler to join paths with the path of the marmot module """
    module_path = os.path.dirname(marmot.representations.tests.__file__)
    resolved = loader.construct_scalar(node)
    return os.path.join(module_path, resolved)

## register the tag handler
yaml.add_constructor('!join', join_with_module_path)


class WMTRepresentationGeneratorTests(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(__file__)
        self.module_path = module_path
        test_config = os.path.join(module_path, 'test_config.yaml')

        with open(test_config, "r") as cfg_file:
            self.config = yaml.load(cfg_file.read())

        self.wmt_target = os.path.join(module_path, 'test_data/EN_ES.tgt_ann.train')
        self.wmt_source = os.path.join(module_path, 'test_data/EN_ES.source.train')
        self.tmp_dir = os.path.join(module_path, 'tmp_dir')

    def tearDown(self):
        if os.path.exists(self.tmp_dir) and os.path.isdir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_load_from_config(self):
        generator = build_object(self.config['representations']['training'][0])
        data_obj = generator.generate()
        self.assertTrue('target' in data_obj)
        self.assertTrue('source' in data_obj)
        self.assertTrue('tags' in data_obj)
        self.assertTrue(len(data_obj['target']) == len(data_obj['source']))
        self.assertTrue(len(data_obj['target']) == len(data_obj['tags']))

    def test_no_saved_files(self):
        generator = WMTRepresentationGenerator(self.wmt_target, self.wmt_source)
        data_obj = generator.generate()
        self.assertTrue('target' in data_obj)
        self.assertTrue('source' in data_obj)
        self.assertTrue('tags' in data_obj)
        self.assertTrue(len(data_obj['target']) == len(data_obj['source']))
        self.assertTrue(len(data_obj['target']) == len(data_obj['tags']))

    def test_save_files(self):
        generator = WMTRepresentationGenerator(self.wmt_target, self.wmt_source, tmp_dir=self.tmp_dir, persist=True)
        data_obj = generator.generate()
        target = os.path.join(self.tmp_dir, 'EN_ES.tgt_ann.train.target')
        tags = os.path.join(self.tmp_dir, 'EN_ES.tgt_ann.train.tags')
        source = os.path.join(self.tmp_dir, 'EN_ES.source.train.txt')
        self.assertTrue(os.path.exists(self.tmp_dir) and os.path.isdir(self.tmp_dir))
        self.assertTrue(os.path.exists(target) and os.path.isfile(target))
        self.assertTrue(os.path.exists(tags) and os.path.isfile(tags))
        self.assertTrue(os.path.exists(source) and os.path.isfile(source))


if __name__ == '__main__':
    unittest.main()

