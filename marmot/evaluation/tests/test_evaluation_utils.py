# test the evaluation functions
import unittest
import os
from marmot.evaluation.evaluation_utils import compare_vocabulary


class TestEvaluate(unittest.TestCase):

    def test_compare_vocabulary(self):
        dataset1 = [['this', 'is', 'sentence', 'number', 'one'], ['another', 'list', 'comes', 'next', '.']]
        dataset2 = [['this', 'is', 'sentence', 'number', 'two'], ['this', 'is', 'sentence', 'number', 'two']]

        comparisons = compare_vocabulary([dataset1, dataset2])
        # 0.4 = fraction of words in sentence 1 covered by sentence 2
        self.assertEqual(comparisons[0]['coverage'], 0.4)
        # 0.8 = fraction of words in sentence 1 covered by sentence 2
        self.assertEqual(comparisons[1]['coverage'], 0.8)


if __name__ == '__main__':
    unittest.main()


