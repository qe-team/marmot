from __future__ import print_function

from subprocess import call
import time
import re
import os
import codecs

from marmot.util.alignments import train_alignments
from marmot.util.force_align import Aligner
from marmot.representations.representation_generator import RepresentationGenerator
from marmot.experiment.import_utils import mk_tmp_dir


class SegmentationRepresentationGenerator(RepresentationGenerator):

    def __init__(self, align_model=None, src_file=None, tg_file=None, lex_prefix=None, tmp_dir=None, moses_dir=None, moses_config=None, workers=1):

        self.tmp_dir = mk_tmp_dir(tmp_dir)
        self.time_stamp = str(time.time())
        self.moses_dir = moses_dir
        self.moses_config = moses_config
        self.workers = workers
        self.lex_prob = lex_prefix

        if align_model is None:
            if src_file is not None and tg_file is not None:
                self.align_model = train_alignments(src_file, tg_file, tmp_dir, align_model=align_model)
            else:
                print("Alignment model not defined, no files for training")
                return
        else:
            self.align_model = align_model

    # write a bash file for the phrase extraction
    def write_command_file(self, data_obj, alignments_file):
        command_name = os.path.join(self.tmp_dir, 'extract_phrases.'+self.time_stamp+'.sh')
        command = open(command_name, 'w')
        # cd to the dir of the script (it doesn't work from any other places because of gzip)
        # TODO: what's the problem with gzip?
        command.write("CUR_DIR=$PWD\ncd %s\n" % self.tmp_dir)
        # extract phrases
        command.write('%s/scripts/generic/extract-parallel.perl 1 split "sort    " %s/bin/extract %s %s %s %s/extract.%s 5 orientation --model wbe-msd --GZOutput\n' % (self.moses_dir, self.moses_dir, data_obj['target_file'], data_obj['source_file'], alignments_file, self.tmp_dir, self.time_stamp))

        # score phrase table halves
        command.write('%s/bin/score extract.%s.sorted.gz %s.f2e %s/phrase-table.%s.half.f2e.gz --GoodTuring  2>> /dev/stderr\n' % (self.moses_dir, self.time_stamp, self.lex_prob, self.tmp_dir, self.time_stamp))
        command.write('%s/bin/score extract.%s.inv.sorted.gz %s.e2f %s/phrase-table.%s.half.e2f.gz --Inverse  2>> /dev/stderr\n' % (self.moses_dir, self.time_stamp, self.lex_prob, self.tmp_dir, self.time_stamp))

        # sort phrase table halves
        command.write('gunzip -c %s/phrase-table.%s.half.f2e.gz | LC_ALL=C sort | gzip -c > %s/phrase-table.%s.half.f2e.sorted.gz\n' % (self.tmp_dir, self.time_stamp, self.tmp_dir, self.time_stamp))
        command.write('gunzip -c %s/phrase-table.%s.half.e2f.gz | LC_ALL=C sort | gzip -c > %s/phrase-table.%s.half.e2f.sorted.gz\n' % (self.tmp_dir, self.time_stamp, self.tmp_dir, self.time_stamp))

        # consolidate halves
        command.write('%s/bin/consolidate %s/phrase-table.%s.half.f2e.sorted.gz %s/phrase-table.%s.half.e2f.sorted.gz /dev/stdout --GoodTuring %s/phrase-table.%s.half.f2e.gz.coc | gzip -c > %s/phrase-table.%s.gz\n' % (self.moses_dir, self.tmp_dir, self.time_stamp, self.tmp_dir, self.time_stamp, self.tmp_dir, self.time_stamp, self.tmp_dir, self.time_stamp))

        command.write('mkdir -p %s/binarized\n' % self.tmp_dir)
        # binarize the phrase table
        command.write('gzip -cd %s/phrase-table.%s.gz | LC_ALL=C sort -T %s/binarized | %s/bin/processPhraseTable -ttable 0 0 - -nscores 4 -out %s/binarized/phrase-table.%s\n' % (self.tmp_dir, self.time_stamp, self.tmp_dir, self.moses_dir, self.tmp_dir, self.time_stamp))
        command.write('rm %s/phrase-table.%s.half.*\n' % (self.tmp_dir, self.time_stamp))
        # return back to where the script was run from
        command.write('cd $CUR_DIR\n')
        command.close()
        phrase_table = os.path.join(self.tmp_dir, 'binarized/phrase-table.{}'.format(self.time_stamp))
        return command_name, phrase_table

    # write Moses config for the current run
    def write_moses_config(self, phrase_table, target_file):
        new_config_name = os.path.join(self.tmp_dir, 'moses.'+self.time_stamp+'.ini')
        new_config = open(new_config_name, 'w')
        constrained = False
        for line in open(self.moses_config):
            if line.startswith("PhraseDictionaryBinary"):
                good_line = [s for s in line.strip().split() if not s.startswith('path')]
                new_config.write("%s path=%s\n" % (' '.join(good_line), phrase_table))
            elif line.startswith("ConstrainedDecoding"):
                new_config.write("ConstrainedDecoding path=%s max-unknowns=-1\n" % target_file)
            elif line.startswith("[weight]") and not constrained:
                new_config.write("ConstrainedDecoding path=%s max-unknowns=-1\n\n" % target_file)
                new_config.write("[weight]\n")
            else:
                new_config.write(line)
        new_config.close()
        return new_config_name

    # src, tg - lists of lists
    # align_file - new file to store the alignments
    # each inner list is a sentence
    def get_alignments(self, src, tg, align_model, align_file):
        alignments = [[[] for j in range(len(tg[i]))] for i in range(len(tg))]
        align_stream = open(align_file, 'w')
        aligner = Aligner(align_model+'.fwd_params', align_model+'.fwd_err', align_model+'.rev_params', align_model+'.rev_err')
        for idx, (src_list, tg_list) in enumerate(zip(src, tg)):
            align_string = aligner.align(' '.join(src_list) + ' ||| ' + ' '.join(tg_list))
            align_stream.write('%s\n' % align_string)
            pairs = align_string.split()
            for p_str in pairs:
                p = p_str.split('-')
                alignments[idx][int(p[1])].append(int(p[0]))
        aligner.close()
        align_stream.close()
        return alignments

    def get_segments(self, data_obj, segmentation_file):
        seg_regexp = re.compile("\|\d+-\d+\|")
        source_segments = []
        target_segments = []
        with codecs.open(segmentation_file, encoding='utf-8') as segmentation:
            for idx, line in enumerate(segmentation):
                # no Moses output for this line - every word is a separate segment
                if line == "\n":
                    source_segments.append([])
                    target_segments.append([(i, i+1) for i in range(len(data_obj['target'][idx]))])
                    continue

                # get source segments
                source_seg_strings = seg_regexp.findall(line)
                source_seg_list = []
                for a_seg in source_seg_strings:
                    a_pair = a_seg.strip('|').split('-')
                    source_seg_list.append((int(a_pair[0]), int(a_pair[1])+1))

                # get target segments
                target_seg_strings = [ll for ll in [l.strip() for l in seg_regexp.split(line)] if ll != '']
                target_seg_list = []
                cur_pos = 0
                for a_seg in target_seg_strings:
                    seg_words = a_seg.split()
                    for (s_w, t_w) in zip(seg_words, data_obj['target'][idx][cur_pos:len(seg_words)]):
                        assert(s_w.lower() == t_w.lower()), "Words don't match at line {}: {} and {}".format(idx, s_w.lower(), t_w.lower())
#                    assert(all([s_w.lower() == t_w.lower() for (s_w, t_w) in zip(seg_words, data_obj['target'][idx][cur_pos:len(seg_words)])])), "Words don't match at line {}: {} and {}".format(idx, )
                    target_seg_list.append((cur_pos, cur_pos+len(seg_words)))
                    cur_pos += len(seg_words)

                # compare source and target segments
                # the number of segments for the source and the target should match
                assert(len(source_seg_list) == len(target_seg_list)), "The numbers of source and target segments don't match: {} and {}".format(len(source_seg_list), len(target_seg_list))

                source_segments.append(source_seg_list)
                target_segments.append(target_seg_list)
        return target_segments, source_segments

    def generate(self, data_obj):
        data_time_stamp = str(time.time())
        if 'target' not in data_obj or 'source' not in data_obj:
            print("No target or source")
        assert(len(data_obj['target']) == len(data_obj['source']))

        # alignments
        alignments_file = os.path.join(self.tmp_dir, 'align.'+self.time_stamp)
        all_alignments = self.get_alignments(data_obj['source'], data_obj['target'], self.align_model, alignments_file)
        data_obj['alignments'] = all_alignments

        # segmentation
        # call Moses phrase extractor
        command, phrase_table = self.write_command_file(data_obj, alignments_file)
        call(['bash', command])

        # call Moses MT
        moses_config = self.write_moses_config(phrase_table, data_obj['target_file'])
        moses_seg_file_name = os.path.join(self.tmp_dir, 'segmentation.'+self.time_stamp+'.'+data_time_stamp)
        moses_seg_file = open(moses_seg_file_name, 'w')
        src = open(data_obj['source_file'])
        call([os.path.join(self.moses_dir, 'bin/moses'), '-f', moses_config, '-v', '0', '-t'], stdin=src, stdout=moses_seg_file)
        moses_seg_file.close()
        src.close()

        data_obj['segmentation'], data_obj['source_segmentation'] = self.get_segments(data_obj, moses_seg_file_name)

        return data_obj
