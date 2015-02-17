# parse a whitespace tokenized file, return an object with the user specified key identifying the parsed data

from parser import Parser
import codecs
from nltk.tokenize import WhitespaceTokenizer


class WhitespaceTokenizedParser(Parser):

    def parse(self, corpus_filename, key):
        assert type(corpus_filename) == str, 'the filename must be a string'
        assert type(key) == str, 'the key must be a string'

        wst = WhitespaceTokenizer()
        with codecs.open(corpus_filename, encoding='utf8') as input:
            corpus = [wst.tokenize(l) for l in input]
        return {key: corpus}
