from __future__ import print_function
import numpy as np
from collections import defaultdict

from marmot.util.alignments import train_alignments
from marmot.util.force_align import Aligner
from marmot.representations.representation_generator import RepresentationGenerator
from marmot.experiment.import_utils import mk_tmp_dir


class AlignmentDoubleRepresentationGenerator(RepresentationGenerator):
    '''
    Extract two types of alignments: 
     - all alignments for every word (list of lists for a sentence)
     - only alignments with the highest confidence are kept for a word (flat list for a sentence)

    The first type is needed for the majority of the features,
    but the PhraseAlignmentFeatureExtractor needs all the possible alignments
    '''

    def __init__(self, lex_file, align_model=None, src_file=None, tg_file=None, tmp_dir=None):

        tmp_dir = mk_tmp_dir(tmp_dir)

        if align_model is None:
            if src_file is not None and tg_file is not None:
                self.align_model = train_alignments(src_file, tg_file, tmp_dir, align_model=align_model)
            else:
                print("Alignment model not defined, no files for training")
                return
        else:
            self.align_model = align_model
        self.lex_prob = self.get_align_prob(lex_file)

    # src, tg - lists of lists
    # each inner list is a sentence
    def get_alignments(self, src, tg, align_model):
        alignments = [[[] for j in range(len(tg[i]))] for i in range(len(tg))]
        aligner = Aligner(align_model+'.fwd_params', align_model+'.fwd_err', align_model+'.rev_params', align_model+'.rev_err')
        for idx, (src_list, tg_list) in enumerate(zip(src, tg)):
            align_string = aligner.align(' '.join(src_list) + ' ||| ' + ' '.join(tg_list))
            pairs = align_string.split()
            for p_str in pairs:
                p = p_str.split('-')
                alignments[idx][int(p[1])].append(int(p[0]))
        aligner.close() 
        return alignments

    # parse lex.f2e file
    # format of self.lex_prob: dictionary of target words
    # every value of the target dictionary is a dictionary of source words
    # every value of the source dictionary is a probability p(target|source):
    # self.lex_prob['el']['he'] = 0.5
    def get_align_prob(self, lex_file):
        lex_dict = defaultdict(lambda: defaultdict(float))
        for line in open(lex_file):
            chunks = line[:-1].decode('utf-8').split()
            assert(len(chunks) == 3), "Wrong format of the lex file: \n{}".format(line)
            val = float(chunks[2])
            lex_dict[chunks[0]][chunks[1]] = val
        return lex_dict

    def generate(self, data_obj):
        if 'alignments' in data_obj:
            print("ALIGNMENTS already exist!")
        if 'target' not in data_obj or 'source' not in data_obj:
            print("No target or source")
        assert(len(data_obj['target']) == len(data_obj['source']))

        all_alignments = self.get_alignments(data_obj['source'], data_obj['target'], self.align_model)
#        print("All alignments: ", all_alignments)
        unique_alignments = []
        for seq_idx, al_sequence in enumerate(all_alignments):
            seq_alignments = []
            for w_idx, al_list in enumerate(al_sequence):
                if len(al_list) > 1:
                    # choose the alignment with the highest probability
#                    print("Multiple alignments: ", al_list)
                    target_word = data_obj['target'][seq_idx][w_idx]
                    source_words = [data_obj['source'][seq_idx][i] for i in al_list]
                    probs = [self.lex_prob[target_word][s] for s in source_words]
#                    print("Probabilities: ", probs)
                    seq_alignments.append(al_list[np.argmax(probs)])
                elif len(al_list) == 0:
                    seq_alignments.append(None)
                elif len(al_list) == 1:
                    seq_alignments.append(al_list[0])
                else:
                    print("Golakteko opasnoste!")
            unique_alignments.append(seq_alignments)

        if 'alignments' not in data_obj:
            data_obj['alignments'] = unique_alignments
        data_obj['alignments_all'] = all_alignments
        return data_obj
