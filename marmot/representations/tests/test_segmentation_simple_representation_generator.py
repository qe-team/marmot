#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
from marmot.representations.segmentation_simple_representation_generator import SegmentationSimpleRepresentationGenerator


class WordQERepresentationGeneratorTests(unittest.TestCase):

    def test_generate(self):
        gen_target = SegmentationSimpleRepresentationGenerator('test_data/tiny.source', 'test_data/tiny.target', 'test_data/tiny.tags', 'test_data/tiny.target_seg', 'target')
        gen_src_tg = SegmentationSimpleRepresentationGenerator('test_data/tiny.source', 'test_data/tiny.target', 'test_data/tiny.tags', 'test_data/tiny.source_target_seg', 'source')
        tg_data = gen_target.generate()

        self.assertListEqual(tg_data['target'], [['el', 'esta', 'un', 'pupil', '.'],
                                                 ['ella', 'sea', 'mi', 'hermano', '.'],
                                                 ['mi', 'gato', 'es', 'amarillo', '.']])
        self.assertListEqual(tg_data['source'], [['he', 'is', 'a', 'pupil', '.'],
                                                 ['she', 'is', 'my', 'sister', '.'],
                                                 ['my', 'cat', 'is', 'smart', '.']])
        self.assertListEqual(tg_data['tags'], [['OK', 'OK', 'OK', 'BAD', 'OK'],
                                               ['OK', 'BAD', 'OK', 'BAD', 'OK'],
                                               ['OK', 'OK', 'BAD', 'BAD', 'OK']])
#        print("Segmentation", tg_data['segmentation'])
#        print("Expected segmentation: ", '[[(0, 1), (1, 3), (3, 5)], [(0, 1), (1, 2), (2, 5)], [(0, 1), (2, 3), (3, 4), (4, 5)]]')
        self.assertListEqual(tg_data['segmentation'], [[(0, 1), (1, 3), (3, 5)],
                                                       [(0, 1), (1, 2), (2, 5)],
                                                       [(0, 2), (2, 3), (3, 4), (4, 5)]])
        self.assertListEqual(tg_data['source_segmentation'], [])

        src_data = gen_src_tg.generate()
#        print(src_data['segmentation'])
#        print('Expected: [[(0, 2), (2, 5)], [], [(0, 1), (1, 3), (3, 4), (4, 5)]]')
#        print(src_data['source_segmentation'])
#        print('Expected: [[(0, 2), (2, 5)], [], [(2, 3), (0, 2), (3, 4), (4, 5)]]')
        self.assertListEqual(src_data['segmentation'], [[(0, 2), (2, 5)], [], [(0, 1), (1, 3), (3, 4), (4, 5)]])
        self.assertListEqual(src_data['source_segmentation'], [[(0, 2), (2, 5)], [], [(2, 3), (0, 2), (3, 4), (4, 5)]])


if __name__ == '__main__':
    unittest.main()
