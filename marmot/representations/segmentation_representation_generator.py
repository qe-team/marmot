import codecs
import re

from marmot.representations.representation_generator import RepresentationGenerator


class SegmentationRepresentationGenerator(RepresentationGenerator):
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

    def __init__(self, source_file, target_file, tags_file, segmentation_file):
        self.data = self.parse_files(source_file, target_file, tags_file, segmentation_file)

    @staticmethod
    def parse_files(source_file, target_file, tags_file, segmentation_file):

        with codecs.open(source_file, encoding='utf8') as source:
            source_lines = [line.split() for line in source]

        with codecs.open(target_file, encoding='utf8') as target:
            target_lines = [line.split() for line in target]

        with codecs.open(tags_file, encoding='utf8') as tags:
            tags_lines = [line.split() for line in tags]

        seg_regexp = re.compile("\|\d+-\d+\|")
        with codecs.open(segmentation_file, encoding='utf-8') as segmentation:
            segments = []
            for line in segmentation:
                seg_strings = seg_regexp.findall(line)
                seg_list = []
                for a_seg in seg_strings:
                    a_pair = a_seg.strip('|').split('-')
                    seg_list.append((int(a_pair[0]), int(a_pair[1])+1))
                segments.append(seg_list)

        return {'target': target_lines, 'source': source_lines, 'tags': tags_lines, 'segmentation': segments}

    def generate(self, data_obj=None):
        return self.data
