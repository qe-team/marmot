import os
from nltk import word_tokenize

from marmot.representations.representation_generator import RepresentationGenerator
from marmot.experiment.import_utils import mk_tmp_dir


class WMTRepresentationGenerator(RepresentationGenerator):

    def _write_to_file(self, filename, lofl):
        a_file = open(filename, 'w')
        for sentence in lofl:
            a_file.write('%s\n' % (' '.join([w.encode('utf-8') for w in sentence])))
        a_file.close()

    def _parse_wmt_to_text(self, wmt_file, wmt_source_file, tmp_dir, persist=False):

        # parse source files
        source_sents = {}
        for line in open(wmt_source_file):
            str_num = line.decode('utf-8').strip().split('\t')
            source_sents[str_num[0]] = word_tokenize(str_num[1])

        # parse target file and write new source, target, and tag files
        target, source, tags = [], [], []
        cur_num = None
        cur_sent, cur_tags = [], []
        for line in open(wmt_file):
            chunks = line[:-1].decode('utf-8').split('\t')
            if chunks[0] != cur_num:
                if len(cur_sent) > 0:
                    # check that the sentence is in source
                    if cur_num in source_sents:
                        source.append(source_sents[cur_num])
                        target.append(cur_sent)
                        tags.append(cur_tags)
                    cur_sent = []
                    cur_tags = []
                cur_num = chunks[0]
            cur_sent.append(chunks[2])
            cur_tags.append(chunks[5])
        # last sentence
        if len(cur_sent) > 0 and cur_num in source_sents:
            source.append(source_sents[cur_num])
            target.append(cur_sent)
            tags.append(cur_tags)

        if persist:
            tmp_dir = mk_tmp_dir(tmp_dir)
            target_file = tmp_dir+'/'+os.path.basename(wmt_file)+'.target'
            tags_file = tmp_dir+'/'+os.path.basename(wmt_file)+'.tags'
            source_file = tmp_dir+'/'+os.path.basename(wmt_source_file)+'.txt'
            self._write_to_file(target_file, target)
            self._write_to_file(source_file, source)
            self._write_to_file(tags_file, tags)

        return {'target': target, 'source': source, 'tags': tags}

    def __init__(self, tg_file, src_file, tmp_dir=None, persist=False):
        self.data = self._parse_wmt_to_text(tg_file, src_file, tmp_dir, persist=persist)

    def generate(self):
        return self.data
