from marmot.util.alignments import train_alignments
from marmot.util.force_align import Aligner
from marmot.representations.representation_generator import RepresentationGenerator
from marmot.experiment.import_utils import mk_tmp_dir


class AlignmentRepresentationGenerator(RepresentationGenerator):

    def __init__(self, align_model=None, src_file=None, tg_file=None, tmp_dir=None):

        tmp_dir = mk_tmp_dir(tmp_dir)

        if align_model is None:
            if src_file is not None and tg_file is not None:
                self.align_model = train_alignments(src_file, tg_file, tmp_dir, align_model=align_model)
            else:
                print("Alignment model not defined, no files for training")
                return
        else:
            self.align_model = align_model

    # src, tg - lists of lists
    # each inner list is a sentence
    def _get_alignments(self, src, tg, align_model):
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

    def generate(self, data_obj):
        if 'target' not in data_obj or 'source' not in data_obj:
            print("No target or source")
        assert(len(data_obj['target']) == len(data_obj['source']))

        all_alignments = self._get_alignments(data_obj['source'], data_obj['target'], self.align_model)

        data_obj['alignments'] = all_alignments
        return data_obj
