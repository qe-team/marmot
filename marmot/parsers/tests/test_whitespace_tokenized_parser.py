import unittest
import os
import codecs

from marmot.parsers.whitespace_tokenized_parser import WhitespaceTokenizedParser


class TestWhitespaceTokenizedParser(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(__file__)
        self.module_path = module_path
        self.test_data = os.path.join(module_path, 'test_data/corpus.en.1000')

    def test_parse(self):
        keyname = 'source'
        data = WhitespaceTokenizedParser().parse(self.test_data, keyname)
        with codecs.open(self.test_data) as lines:
            line_count = sum(1 for l in lines)

        self.assertTrue(keyname in data)
        self.assertEqual(len(data[keyname]), line_count)


if __name__ == '__main__':
    unittest.main()
