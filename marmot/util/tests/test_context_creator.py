# TODO: stub - implement

import unittest, os
from marmot.util.corpus_context_creator import CorpusContextCreator

class TestRunExperiment(unittest.TestCase):

    def setUp(self):
        module_path = os.path.dirname(__file__)
        self.module_path = module_path
        # create the set of tokens we're interested in
        self.important_tokens = set(['and', 'the'])

        # create a testing dataset
        test_contexts = [ \
            {'index': 17, 'token': u'and', 'tag': 1, 'target': [u'so', u',', u'ladies', u'and', u'gentlemen', u',', u'i', u'should', u'like', u'in', u'a', u'moment', u'to', u'return', u'to', u'the', u'role', u'and', u'structure', u'of', u'the', u'guidelines', u'before', u'mentioning', u'the', u'principal', u'comments', u'and', u'criticisms', u'that', u'you', u',', u'mrs', u'schroedter', u',', u'and', u'the', u'various', u'members', u'of', u'this', u'house', u',', u'have', u'made', u'.'], 'source': None}, \
            {'index': 3, 'token': u'and', 'tag': 1, 'target': [u'genuine', u'structural', u'reforms', u'and', u'a', u'competition', u'-', u'friendly', u'taxation', u'policy', u'are', u'the', u'cornerstones', u'of', u'a', u'successful', u'economic', u'base', u'.'], 'source': None}, \
            {'index': 23, 'token': u'and', 'tag': 1, 'target': [u'even', u'the', u'accumulation', u'of', u'money', u'from', u'the', u'cohesion', u'funds', u'and', u'the', u'structural', u'funds', u'has', u'failed', u'to', u'have', u'the', u'desired', u'effect', u'in', u'all', u'regions', u'and', u'countries', u'.'], 'source': None}, \
            {'index': 34, 'token': u'and', 'tag': 1, 'target': [u'the', u'commission', u'report', u'is', u'essentially', u'a', u'descriptive', u'report', u'detailing', u'the', u'development', u'of', u'state', u'aid', u'in', u'the', u'manufacturing', u'sector', u'and', u'certain', u'other', u'sectors', u',', u'according', u'to', u'various', u'typologies', u',', u'such', u'as', u'the', u'method', u'of', u'financing', u'and', u'the', u'objectives', u'pursued', u'.'], 'source': None}, \
            {'index': 5, 'token': u'the', 'tag': 1, 'target': [u'finally', u',', u'we', u'ask', u'that', u'the', u'commission', u'ensures', u'that', u'structural', u'fund', u'monies', u'are', u'spent', u'in', u'a', u'way', u'which', u'is', u'transparent', u'.'], 'source': None}, \
            {'index': 10, 'token': u'the', 'tag': 1, 'target': [u'that', u'way', u'the', u'much', u'-', u'trumpeted', u'need', u'for', u'transparency', u'in', u'the', u'use', u'of', u'these', u'funds', u'and', u'the', u'temptation', u'to', u'draw', u'unnecessarily', u'in', u'the', u'longer', u'term', u'on', u'the', u'local', u'tax', u'base', u'in', u'areas', u'where', u'such', u'projects', u'are', u'located', u'will', u'be', u'diminished', u'and', u'the', u'european', u'parliament', u'will', u'show', u'how', u'seriously', u'it', u'takes', u'the', u'need', u'for', u'such', u'reform', u'.'], 'source': None}, \
        ]

        # build a corpus context creator from our contexts
        self.corpus_cc = CorpusContextCreator(test_contexts)

    def test_get_contexts(self):
        and_contexts = self.corpus_cc.get_contexts('and')
        # we initialized with num_contexts=10
        self.assertTrue(len(and_contexts) == 4)

if __name__ == '__main__':
    unittest.main()
