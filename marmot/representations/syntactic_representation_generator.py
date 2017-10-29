from marmot.util.extract_syntactic_features import call_stanford, call_parzu, parse_xml, parse_conll
from marmot.representations.representation_generator import RepresentationGenerator


class SyntacticRepresentationGenerator(RepresentationGenerator):

    def __init__(self, tmp_dir, reverse=False):
        self.tmp_dir = tmp_dir
        self.reverse = reverse

    def generate(self, data_obj):
        if self.reverse:
            parsed_src = call_parzu(data_obj['source_file'], self.tmp_dir)
            parsed_tg = call_stanford(data_obj['target_file'], self.tmp_dir)
            sentences_tg = parse_xml(parsed_tg)
            sentences_src = parse_conll(parsed_src)
        else:
            parsed_src = call_stanford(data_obj['source_file'], self.tmp_dir)
            parsed_tg = call_parzu(data_obj['target_file'], self.tmp_dir)
            sentences_src = parse_xml(parsed_src)
            sentences_tg = parse_conll(parsed_tg)
        data_obj['target_dependencies'] = [sent['dependencies'] for sent in sentences_tg]
        data_obj['source_dependencies'] = [sent['dependencies'] for sent in sentences_src]
        data_obj['target_synt_pos'] = [sent['pos'] for sent in sentences_tg]
        data_obj['source_synt_pos'] = [sent['pos'] for sent in sentences_src]
#        data_obj['target_root'] = [sent['root'] for sent in sentences_tg]
#        data_obj['source_root'] = [sent['root'] for sent in sentences_src]
        del data_obj['target_file']
        del data_obj['source_file']
        return data_obj
