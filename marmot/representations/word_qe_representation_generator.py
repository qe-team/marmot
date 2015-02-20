import codecs

from marmot.representations.representation_generator import RepresentationGenerator
from nltk.tokenize import WhitespaceTokenizer
from marmot.experiment.import_utils import mkdir


class WordQERepresentationGenerator(RepresentationGenerator):
    '''
    The standard word-level format: 3 files, source, target, tags, one line per file, whitespace tokenized
    '''

    def __init__(self, source_file, target_file, tags_file):
        self.tokenizer = WhitespaceTokenizer()
        self.data = self.parse_files(source_file, target_file, tags_file)

    def parse_files(self, source_file, target_file, tags_file):

        with codecs.open(source_file, encoding='utf8') as source:
            source_lines = [self.tokenizer.tokenize(line) for line in source]

        with codecs.open(target_file, encoding='utf8') as target:
            target_lines = [self.tokenizer.tokenize(line) for line in target]

        with codecs.open(tags_file, encoding='utf8') as tags:
            tags_lines = [self.tokenizer.tokenize(line) for line in tags]

        return {'target': target_lines, 'source': source_lines, 'tags': tags_lines}

    def generate(self, data_obj=None):
        return self.data
