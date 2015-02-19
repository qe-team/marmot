from __future__ import print_function

from nltk import word_tokenize
from goslate import Goslate

from marmot.representations.representation_generator import RepresentationGenerator


class GoogleTranslateRepresentationGenerator(RepresentationGenerator):
    '''
    Generate pseudoreference with Google Translate
    '''

    # <lang> -- target language
    def __init__(self, lang='en'):
        self.lang = lang
        self.gs = Goslate()

    def generate(self, data_obj):
        if 'source' not in data_obj:
            print('No source for pseudo-reference generation')
            return data_obj

        references = []
        try:
            for ref in self.gs.translate([' '.join(sentence) for sentence in data_obj['source']], self.lang):
                references.append(word_tokenize(ref))
        # TODO: might it be some other error?
        except:
            print('Network error, no pseudo-reference is generated')
            return data_obj

        data_obj['pseudo-reference'] = references
        return data_obj
