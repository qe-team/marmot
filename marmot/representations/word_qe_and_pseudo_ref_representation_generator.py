import codecs
from nltk import wordpunct_tokenize

from marmot.representations.representation_generator import RepresentationGenerator


class WordQEAndPseudoRefRepresentationGenerator(RepresentationGenerator):
    '''
    Generate the standard word-level format: 3 files, source, target, tags, one line per file, whitespace tokenized
    Also add the un-tokenized pseudo-references to the dataset
    '''

    def __init__(self, source_file, target_file, tags_file, pseudo_ref_file):
        self.data = self.parse_files(source_file, target_file, tags_file, pseudo_ref_file)

    @staticmethod
    def parse_files(source_file, target_file, tags_file, pseudo_ref_file):

        with codecs.open(source_file, encoding='utf8') as source:
            source_lines = [line.split() for line in source]

        with codecs.open(target_file, encoding='utf8') as target:
            target_lines = [line.split() for line in target]

        with codecs.open(tags_file, encoding='utf8') as tags:
            tags_lines = [line.split() for line in tags]

        with codecs.open(pseudo_ref_file, encoding='utf8') as pseudo_ref:
            pseudo_ref_lines = [wordpunct_tokenize(line.strip()) for line in pseudo_ref]

        assert len(source_lines) == len(target_lines) == len(tags_lines) == len(pseudo_ref_lines)

        return {'target': target_lines, 'source': source_lines, 'tags': tags_lines, 'pseudo_ref': pseudo_ref_lines}

    def generate(self, data_obj=None):
        return self.data
