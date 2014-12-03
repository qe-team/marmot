# get the utf8 words from a text file

from nltk.tokenize import word_tokenize
import codecs

def get_tokens(filename):
    with codecs.open(filename, encoding='utf8') as input:
        all_lines = ' '.join(input.read().splitlines())
        for word in word_tokenize(all_lines):
            yield word


