#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
from marmot.representations.word_qe_representation_generator import WordQERepresentationGenerator
from marmot.representations.alignment_representation_generator import AlignmentRepresentationGenerator


class WordQERepresentationGeneratorTests(unittest.TestCase):

    def test_generate(self):
        main_generator = WordQERepresentationGenerator('test_data/tiny.source_align', 'test_data/tiny.target', 'test_data/tiny.tags')
        align_generator = AlignmentRepresentationGenerator('/export/data/varvara/europarl-sys/english_spanish/model/lex.1.f2e', align_model='/export/data/varvara/my_marmot/my_marmot/experiment/tiny_test/europarl_align_model')

        data = main_generator.generate()
        data = align_generator.generate(data)
        self.assertListEqual(data['alignments'], [[0, 1, 2, 3, 4], [None, 0, None, 1, 2], [0, 1, 2, 4, 5]])


if __name__ == '__main__':
    unittest.main()
