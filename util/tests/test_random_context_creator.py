import unittest, os
from marmot.util.random_context_creator import RandomContextCreator

class TestRunExperiment(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(__file__)
        self.module_path = module_path
        self.target_vocabulary = set(['one', 'two', 'three', 'four', 'five'])
        # def __init__(self, word_list, length_bounds=[6,12], tagset=set([0])):
        self.random_cc = RandomContextCreator(self.target_vocabulary, num_contexts=10)

    def test_get_contexts(self):
        a_context = self.random_cc.get_contexts('apple')
        print(a_context)
        # we initialized with num_contexts=10
        self.assertTrue(len(a_context) == 10)

if __name__ == '__main__':
    unittest.main()
