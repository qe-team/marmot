from subprocess import Popen
import os
import time

from marmot.representations.representation_generator import RepresentationGenerator
from marmot.experiment.import_utils import mk_tmp_dir


class POSRepresentationGenerator(RepresentationGenerator):

    def _get_random_name(self, suffix=''):
        return 'tmp_'+suffix+str(time.time())

    def _get_pos_tagging(self, src, tagger, par_file, tmp_dir):
        # tokenize and add the sentence end marker
        # tokenization is done with nltk
        tmp_tokenized_name = os.path.join(tmp_dir, self._get_random_name('tok'))
        tmp_tok = open(tmp_tokenized_name, 'wr+')
        for words in src:
            tmp_tok.write('%s\nSentenceEndMarker\n' % '\n'.join([w.encode('utf-8') for w in words]))
        tmp_tok.seek(0)

        # pass to tree-tagger
        tmp_tagged_name = os.path.join(tmp_dir, self._get_random_name('tag'))
        tmp_tagged = open(tmp_tagged_name, 'wr+')
        tagger_call = Popen([tagger, '-token', par_file], stdin=tmp_tok, stdout=tmp_tagged)
        tagger_call.wait()
        tmp_tagged.seek(0)

        # remove sentence markers, restore sentence structure
        output = []
        cur_sentence = []
        for line in tmp_tagged:
            word_tag = line[:-1].decode('utf-8').strip().split('\t')
            # each string has to be <word>\t<tag>
            # TODO: if it's not of this format, it could be the end of sequence (empty string) or an error
            if len(word_tag) != 2:
                continue
            if word_tag[0] == 'SentenceEndMarker':
                output.append(cur_sentence)
                cur_sentence = []
            else:
                cur_sentence.append(word_tag[1])
        tmp_tok.close()
        tmp_tagged.close()

        # delete all temporary files
        os.remove(tmp_tokenized_name)
        os.remove(tmp_tagged_name)

        return output

    # <tagger> -- path to tree-tagger
    # <parameters> -- parameters of tree-tagger
    # <data_label> -- which data should be tagged ('source' or 'target')
    def __init__(self, tagger, parameters, data_label, tmp_dir=None):
        self.tmp_dir = mk_tmp_dir(tmp_dir)

        self.tagger = tagger
        self.parameters = parameters
        self.data = data_label

    def generate(self, data_obj):
        data_obj[self.data+'_pos'] = self._get_pos_tagging(data_obj[self.data], self.tagger, self.parameters, self.tmp_dir)
        return data_obj
