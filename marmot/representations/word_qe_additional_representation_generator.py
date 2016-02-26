import codecs
import sys
from marmot.representations.representation_generator import RepresentationGenerator


class WordQEAdditionalRepresentationGenerator(RepresentationGenerator):
    '''
    The standard word-level format + additional file(s): filename saved
    '''

    def __init__(self, source_file, target_file, tags_file, additional_files=None, additional_names=None):
        self.data = self.parse_files(source_file, target_file, tags_file, additional_files=additional_files, additional_names=additional_names)

    @staticmethod
    def parse_files(source_file, target_file, tags_file, additional_files=None, additional_names=None):

        with codecs.open(source_file, encoding='utf8') as source:
            source_lines = [line.split() for line in source]

        with codecs.open(target_file, encoding='utf8') as target:
            target_lines = [line.split() for line in target]

        with codecs.open(tags_file, encoding='utf8') as tags:
            tags_lines = [line.split() for line in tags]

        data_obj = {'target': target_lines, 'source': source_lines, 'tags': tags_lines}

        if additional_files is not None and additional_names is not None:
            for add_file, add_name in zip(additional_files, additional_names):
                data_obj[add_name] = add_file

        return data_obj

    def generate(self, data_obj=None):
        return self.data
