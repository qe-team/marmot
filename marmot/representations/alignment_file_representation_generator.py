from __future__ import print_function
import numpy as np
import sys
from collections import defaultdict
from marmot.representations.representation_generator import RepresentationGenerator


class AlignmentFileRepresentationGenerator(RepresentationGenerator):
    '''
    Get alignments from file
    '''

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

    def get_alignments(self, align_file, target_lines):
        alignments = []
        cnt = 0
        for words, line in zip(target_lines, open(align_file)):
            cnt += 1
            cur_align_dict = defaultdict(list)
            for pair in line.strip('\n').split():
                pair = pair.split('-')
                cur_align_dict[int(pair[1])].append(int(pair[0]))
            cur_align = []
            for i in range(len(words)):
                cur_align.append(cur_align_dict[i])
            alignments.append(cur_align)
        return alignments

    def __init__(self, lex_file):
        self.lex_prob = self.get_align_prob(lex_file)

    def generate(self, data_obj):
        all_alignments = self.get_alignments(data_obj['alignments_file'], data_obj['target'])

        unique_alignments = []
        for seq_idx, al_sequence in enumerate(all_alignments):
            seq_alignments = []
            for w_idx, al_list in enumerate(al_sequence):
                if len(al_list) > 1:
                    try:
                        target_word = data_obj['target'][seq_idx][w_idx]
                        source_words = [data_obj['source'][seq_idx][i] for i in al_list]
                    except IndexError:
                        print("TArget:", data_obj['target'][seq_idx])
                        print("Source:", data_obj['source'][seq_idx])
                        print("Target: {} seq, needed {}. {} words, needed {}".format(len(data_obj['target']), seq_idx, len(data_obj['target'][seq_idx]), w_idx))
                        sys.exit()
                    probs = [self.lex_prob[target_word][s] for s in source_words]
                    seq_alignments.append(al_list[np.argmax(probs)])
                elif len(al_list) == 0:
                    seq_alignments.append(None)
                elif len(al_list) == 1:
                    seq_alignments.append(al_list[0])
                else:
                    print("Golakteko opasnoste!")
            unique_alignments.append(seq_alignments)
        data_obj['alignments'] = unique_alignments

        # remove alignments file, we don't need it any more
        del data_obj['alignments_file']
        return data_obj
