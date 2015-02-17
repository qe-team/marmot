import unittest, os
from marmot.preprocessing.words_from_file import get_tokens

class WordsFromFileTests(unittest.TestCase):
    def setUp(self):
        self.interesting_tokens = set(['the','it'])
        module_path = os.path.dirname(__file__)
        self.corpus_file = os.path.join(module_path, 'test_data/corpus.en.1000')

    def test_words_from_file(self):
        token_generator = get_tokens(self.corpus_file)
        token_set = set(token_generator)
        self.assertTrue(len(token_set) > 0)
        for word in token_set:
            self.assertTrue(type(word) == unicode)


if __name__ == '__main__':
    unittest.main()
