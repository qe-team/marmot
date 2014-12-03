#!/usr/bin/env python
#encoding: utf-8

'''
@author: Chris Hokamp
@contact: chris.hokamp@gmail.com
'''

from nltk.tokenize import word_tokenize
import unittest
from marmot.util import ngram_window_extractor


class TestNgramFeatureExtractor(unittest.TestCase):

    def test_extract_window(self):
        sen_str = 'this is a test sentence.'
        sen = word_tokenize(sen_str.lower())
        test_token = 'this'
        window = ngram_window_extractor.extract_window(sen, test_token)
        self.assertListEqual(window, ['_START_', 'this', 'is'], 'A window starting with the first token should be correct')

        sen2_str = 'this is a test sentence.'
        sen2 = word_tokenize(sen2_str.lower())
        test_token2 = 'is'
        window2 = ngram_window_extractor.extract_window(sen2, test_token2)
        self.assertListEqual(window2, ['this', 'is', 'a'], 'A window starting with the second token should be correct')

    def test_left_context(self):
        sen_str = 'this is a test sentence.'
        sen = word_tokenize(sen_str.lower())
        test_token = 'is'
        left_context = ngram_window_extractor.left_context(sen, test_token, context_size=3)
        self.assertListEqual(left_context, ['_START_', '_START_', 'this'], 'left_context should prepend _START_ tokens')

    def test_right_context(self):
        sen_str = 'this is a test sentence.'
        sen = word_tokenize(sen_str.lower())
        test_token = 'sentence'
        right_context = ngram_window_extractor.right_context(sen, test_token, context_size=3)
        self.assertListEqual(right_context, ['.', '_END_', '_END_'], 'right_context should append _END_ tokens')


if __name__ == '__main__':
    unittest.main()
