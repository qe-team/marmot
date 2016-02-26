import codecs
import re

from marmot.representations.representation_generator import RepresentationGenerator


class SegmentationSimpleRepresentationGenerator(RepresentationGenerator):
    '''
    Source, target, tags, segmentation files, one line per file, whitespace tokenized
    Segmentation file -- can be Moses output with phrase segmentation (with '-t' option)
        or just have the information on segments.
    Segments have to be in the form |i-j| where i is the index of the first word in segment,
                                                j -- the last word in segment.
    Examples of acceptable segmentation formats:
       (1) he is |0-1| a good |2-3| ukulele |4-4| player |5-5| . |6-6|
       (2) |0-1| |2-3| |4-4| |5-5| |6-6|
    in the format (1) the words can be from the source lang (or any other), only numbers matter
    the format (2) is the same as (1) but with no words
    they are parsed in the same way: just substrings '|i-j|' are extracted
    '''

    def __init__(self, source_file, target_file, tags_file, segmentation_file, segmentation_numbers='target'):
        '''
        Parameters:
         - <source_file>
         - <target_file>
         - <tags_file>
         - <segmentation_file>. Acceptable formats:
                (1) he is |0-1| a good |2-3| ukulele |4-4| player |5-5| . |6-6|
                (2) |0-1| |2-3| |4-4| |5-5| |6-6|
         - <segmentation_numbers> - 'source' or 'target', default - 'target'
                the side whose segment borders are denoted with numbers in <segmentation_file>
                If the <segmentation_file> has format (1),
                then <segmentation_numbers> has to be 'target'
        '''
        self.data = self.parse_files(source_file, target_file, tags_file, segmentation_file, segmentation_numbers)

    @staticmethod
    def parse_files(source_file, target_file, tags_file, segmentation_file, segmentation_numbers):

        with codecs.open(source_file, encoding='utf8') as source:
            source_lines = [line.split() for line in source]

        with codecs.open(target_file, encoding='utf8') as target:
            target_lines = [line.split() for line in target]

        with codecs.open(tags_file, encoding='utf8') as tags:
            tags_lines = [line.split() for line in tags]

        seg_regexp = re.compile("\|\d+-\d+\|")
        with codecs.open(segmentation_file, encoding='utf-8') as segmentation:
            segments, source_segments = [], []
            if segmentation_numbers == 'target':
                for line in segmentation:
                    seg_strings = seg_regexp.findall(line)
                    seg_list = []
                    for a_seg in seg_strings:
                        a_pair = a_seg.strip('|').split('-')
                        seg_list.append((int(a_pair[0]), int(a_pair[1])+1))
                    # segments need to be sorted, because they could be reordered during decoding:
                    # he is |3-4| a good |0-1| ukulele player |2-2| . |5-5|
                    # seg_list == [(3, 5), (0, 2), (2, 3), (5, 6)]
                    # sorted(seg_list) == [(0, 2), (2, 3), (3, 5), (5, 6)]
                    seg_list = sorted(seg_list)
                    new_seg_list = []
                    prev = 0
                    for s in seg_list:
                        # end of previous segment doesn't match the beginning of the current segment
                        # this means that one or more of the words wasn't included into the segmentation
                        # have to be added as a separate segment
                        if s[0] != prev:
                            new_seg_list.append((prev, s[0]))
                        new_seg_list.append(s)
                        prev = s[1]
                    segments.append(sorted(new_seg_list))
                return {'target': target_lines, 'source': source_lines, 'tags': tags_lines, 'segmentation': segments}
            elif segmentation_numbers == 'source':
                for line in segmentation:
                    if line == '\n':
                        segments.append([])
                        source_segments.append([])
                        continue
                    seg_strings = seg_regexp.split(line[:-1])
                    source_seg_strings = seg_regexp.findall(line)
                    seg_lengths = [len(a_seg.strip().split()) for a_seg in seg_strings if len(a_seg.split()) > 0]
                    # segmentation doesn't exits for the sentence
                    if len(seg_lengths) == 0:
                        segments.append([])
                        source_segments.append([])
                        continue
                    start = 0
                    seg_list = []
                    for a_len in seg_lengths:
                        seg_list.append((start, start + a_len))
                        start += a_len
                    source_seg_list = []
                    for a_seg in source_seg_strings:
                        a_pair = a_seg.strip('|').split('-')
                        source_seg_list.append((int(a_pair[0]), int(a_pair[1])+1))
                    segments.append(seg_list)
                    # here segments mustn't be sorted, to keep the correspondence between the source and the target
                    source_segments.append(source_seg_list)
                return {'target': target_lines, 'source': source_lines, 'tags': tags_lines, 'segmentation': segments, 'source_segmentation': source_segments}
            else:
                print("Unknown segmentation numbers value: {}".format(segmentation_numbers))


    def generate(self, data_obj=None):
        return self.data
