from marmot.representations.representation_generator import RepresentationGenerator
import codecs


class SegmentationDoubleRepresentationGenerator(RepresentationGenerator):
    '''
    Both source and target are already segmented with '||'
    '''

    def get_segments_from_line(self, line):
        seg = line.strip('\n').split(' || ')
        cur_words, cur_seg = [], []
        cur_pos = 0
        for seg in line.strip('\n').split(' || '):
            seg_split = seg.split()
            cur_words.extend(seg_split)
            cur_seg.append((cur_pos, cur_pos + len(seg_split)))
            cur_pos += len(seg_split)
        return cur_words, cur_seg

    def parse_files(self, source_file, target_file, tags_file, word_align_file):
        # extract source segments
        with codecs.open(source_file, encoding='utf8') as source:
            source_words, source_segments = [], []
            for line in source:
                cur_words, cur_seg = self.get_segments_from_line(line)
                source_words.append(cur_words)
                source_segments.append(cur_seg)

        # extract target segments
        with codecs.open(target_file, encoding='utf8') as target:
            target_words, target_segments = [], []
            for line in target:
                cur_words, cur_seg = self.get_segments_from_line(line)
                target_words.append(cur_words)
                target_segments.append(cur_seg)

        with codecs.open(tags_file, encoding='utf8') as tags:
            phrase_tags = [line.split() for line in tags]

        return {'segmentation': target_segments, 'source_segmentation': source_segments, 'source': source_words, 'target': target_words, 'alignments_file': word_align_file, 'tags': phrase_tags}

    def __init__(self, source_file, target_file, tags_file, word_align_file):
        self.data = self.parse_files(source_file, target_file, tags_file, word_align_file)

    def generate(self, data_obj=None):
        return self.data
