#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import os
import yaml

import marmot
from marmot.representations.word_qe_representation_generator import WordQERepresentationGenerator
from marmot.experiment.import_utils import build_object


def join_with_module_path(loader, node):
    """ define custom tag handler to join paths with the path of the marmot module """
    module_path = os.path.dirname(marmot.representations.tests.__file__)
    resolved = loader.construct_scalar(node)
    return os.path.join(module_path, resolved)

## register the tag handler
yaml.add_constructor('!join', join_with_module_path)

class WordQERepresentationGeneratorTests(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(__file__)
        self.module_path = module_path
        test_config = os.path.join(module_path, 'test_config.yaml')

        with open(test_config, "r") as cfg_file:
            self.config = yaml.load(cfg_file.read())

        self.target_file = os.path.join(module_path, 'test_data/dev.target')
        self.source_file = os.path.join(module_path, 'test_data/dev.source')
        self.tags_file = os.path.join(module_path, 'test_data/dev.target.tags')

    def test_generator(self):
        generator = WordQERepresentationGenerator(self.source_file, self.target_file, self.tags_file)
        data_obj = generator.generate()
        self.assertTrue('target' in data_obj)
        self.assertTrue('source' in data_obj)
        self.assertTrue('tags' in data_obj)
        self.assertTrue(len(data_obj['target']) == len(data_obj['source']) == len(data_obj['tags']))
        self.assertTrue(len(data_obj['target']) == len(data_obj['tags']))

    def test_load_from_config(self):
        generator = build_object(self.config['representations']['training'][1])
        data_obj = generator.generate()
        self.assertTrue('target' in data_obj)
        self.assertTrue('source' in data_obj)
        self.assertTrue('tags' in data_obj)
        self.assertTrue(len(data_obj['target']) == len(data_obj['source']) == len(data_obj['tags']))
        self.assertTrue(len(data_obj['target']) == len(data_obj['tags']))

    # TODO: test that tokenization happens like we expect


if __name__ == '__main__':
    unittest.main()
