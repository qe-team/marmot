#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
from marmot.experiment.context_utils import *


class WordQERepresentationGeneratorTests(unittest.TestCase):
    
    def setUp(self):
        self.repr_dict = {'target': ['un', 'nino', 'el', 'perro', 'bati'], 'source': ['a', 'boy', 'hits', 'a', 'dog'], 'tags': ['OK', 'OK', 'BAD', 'BAD', 'OK'], 'segmentation': [(0, 2), (2, 4), (4, 5)], 'source_segmentation': [(0, 2), (3, 5), (2, 3)]}
        self.repr_dict_no_src = {'target': ['un', 'nino', 'bati', 'el', 'perro'], 'source': ['a', 'boy', 'hits', 'a', 'dog'], 'tags': ['OK', 'OK', 'BAD', 'BAD', 'OK'], 'segmentation': [(0, 1), (1, 4), (4, 5)], 'alignments': [[], [0, 1], [2], [3], [3, 4]]}
        self.repr_dict_no_align = {'target': ['un', 'nino', 'el', 'perro', 'bati'], 'source': ['a', 'boy', 'hits', 'a', 'dog'], 'tags': ['OK', 'OK', 'BAD', 'BAD', 'OK'], 'segmentation': [(0, 2), (2, 4), (4, 5)]}
        self.repr_dict_empty_align = {'target': ['un', 'nino', 'el', 'perro', 'bati'], 'source': ['a', 'boy', 'hits', 'a', 'dog'], 'tags': ['OK', 'OK', 'BAD', 'BAD', 'OK'], 'segmentation': [(0, 2), (2, 4), (4, 5)], 'alignments': [[], [], [], [4], []]}

    def test_error_based_segmentation(self):
        new_seg = error_based_segmentation(self.repr_dict_no_src)
        self.assertListEqual(new_seg, [(0, 1), (1, 2), (2, 4), (4, 5)])

    def test_error_based_segmentation_src(self):
        try:
            create_context_phrase(self.repr_dict, unambiguous=True)
            self.fail("Didn't raise the exception")
        except AssertionError as ae:
            self.assertEqual(ae.args[0], "Error-based segmentation of target can't be performed if source segmentation exists -- after re-segmentation source and target segments won't match")
        except Exception, e:
            self.fail("Raised the wrong exception: ", e.args)

    def test_context_phrase(self):
        contexts = create_context_phrase(self.repr_dict)
        self.assertDictEqual(contexts[0], {'token': ['un', 'nino'], 'index': (0, 2), 'source_token': ['a', 'boy'], 'source_index': (0, 2), 'tag': 'OK', 'target': ['un', 'nino', 'el', 'perro', 'bati'], 'source': ['a', 'boy', 'hits', 'a', 'dog']})
        self.assertDictEqual(contexts[1], {'token': ['el', 'perro'], 'index': (2, 4), 'source_token': ['a', 'dog'], 'source_index': (3, 5), 'tag': 'BAD', 'target': ['un', 'nino', 'el', 'perro', 'bati'], 'source': ['a', 'boy', 'hits', 'a', 'dog']})
        self.assertDictEqual(contexts[2], {'token': ['bati'], 'index': (4, 5), 'source_token': ['hits'], 'source_index': (2, 3), 'tag': 'OK', 'target': ['un', 'nino', 'el', 'perro', 'bati'], 'source': ['a', 'boy', 'hits', 'a', 'dog']})

    def test_context_phrase_no_align(self):
        contexts = create_context_phrase(self.repr_dict_no_align)
        self.assertDictEqual(contexts[0], {'token': ['un', 'nino'], 'index': (0, 2), 'source_token': [], 'source_index': (), 'tag': 'OK', 'target': ['un', 'nino', 'el', 'perro', 'bati'], 'source': ['a', 'boy', 'hits', 'a', 'dog']})
        self.assertDictEqual(contexts[1], {'token': ['el', 'perro'], 'index': (2, 4), 'source_token': [], 'source_index': (), 'tag': 'BAD', 'target': ['un', 'nino', 'el', 'perro', 'bati'], 'source': ['a', 'boy', 'hits', 'a', 'dog']})
        self.assertDictEqual(contexts[2], {'token': ['bati'], 'index': (4, 5), 'source_token': [], 'source_index': (), 'tag': 'OK', 'target': ['un', 'nino', 'el', 'perro', 'bati'], 'source': ['a', 'boy', 'hits', 'a', 'dog']})

    def test_context_phrase_empty_align(self):
        contexts = create_context_phrase(self.repr_dict_empty_align)
        self.assertDictEqual(contexts[0], {'token': ['un', 'nino'], 'index': (0, 2), 'source_token': [], 'source_index': (), 'tag': 'OK', 'target': ['un', 'nino', 'el', 'perro', 'bati'], 'source': ['a', 'boy', 'hits', 'a', 'dog'], 'alignments': [[], [], [], [4], []]})
        self.assertDictEqual(contexts[1], {'token': ['el', 'perro'], 'index': (2, 4), 'source_token': ['dog'], 'source_index': (4, 5), 'tag': 'BAD', 'target': ['un', 'nino', 'el', 'perro', 'bati'], 'source': ['a', 'boy', 'hits', 'a', 'dog'], 'alignments': [[], [], [], [4], []]})
        self.assertDictEqual(contexts[2], {'token': ['bati'], 'index': (4, 5), 'source_token': [], 'source_index': (), 'tag': 'OK', 'target': ['un', 'nino', 'el', 'perro', 'bati'], 'source': ['a', 'boy', 'hits', 'a', 'dog'], 'alignments': [[], [], [], [4], []]})

    def test_context_phrase_unambiguous_segm(self):
        pass

    def test_context_phrase_alignments(self):
        contexts = create_context_phrase(self.repr_dict_no_src)
        self.assertDictEqual(contexts[0], {'target': ['un', 'nino', 'bati', 'el', 'perro'], 'source': ['a', 'boy', 'hits', 'a', 'dog'], 'alignments': [[], [0, 1], [2], [3], [3, 4]], 'token': ['un'], 'index': (0, 1), 'source_token': [], 'source_index': (), 'tag': 'OK'})
        self.assertDictEqual(contexts[1], {'target': ['un', 'nino', 'bati', 'el', 'perro'], 'source': ['a', 'boy', 'hits', 'a', 'dog'], 'alignments': [[], [0, 1], [2], [3], [3, 4]], 'token': ['nino', 'bati', 'el'], 'index': (1, 4), 'source_token': ['a', 'boy', 'hits', 'a'], 'source_index': (0, 4), 'tag': 'BAD'})
        self.assertDictEqual(contexts[2], {'target': ['un', 'nino', 'bati', 'el', 'perro'], 'source': ['a', 'boy', 'hits', 'a', 'dog'], 'alignments': [[], [0, 1], [2], [3], [3, 4]], 'token': ['perro'], 'index': (4, 5), 'source_token': ['a', 'dog'], 'source_index': (3, 5), 'tag': 'OK'})


if __name__ == '__main__':
    unittest.main()
