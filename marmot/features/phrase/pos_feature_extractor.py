from __future__ import division

from collections import defaultdict
import logging

from marmot.features.feature_extractor import FeatureExtractor

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('testlogger')


# collections of content tags for some languages
def get_tags(lang):
    content, verbs, nouns = defaultdict(list), defaultdict(list), defaultdict(list)
    content['english'] = ['JJ', 'RB', 'NN', 'VB']
    content['spanish'] = ['ADJ', 'ADV', 'NC', 'NMEA', 'NMON', 'NP', 'VL']
    verbs['english'] = ['VB']
    verbs['spanish'] = ['VL']
    nouns['english'] = ['NN']
    nouns['spanish'] = ['NC', 'NMEA', 'NMON', 'NP']
    return content[lang], nouns[lang], verbs[lang]


class POSFeatureExtractor(FeatureExtractor):

    def belongs_to(self, word_tag, category):
        for tag in category:
            if word_tag.startswith(tag):
                return True
        return False

    def __init__(self, lang_src, lang_tg):
        self.content_src, self.nouns_src, self.verbs_src = get_tags(lang_src)
        self.content_tg, self.nouns_tg, self.verbs_tg = get_tags(lang_tg)
        if len(self.content_src) == 0:
            logger.warn("No POS lists for the language {}".format(lang_src))
        if len(self.content_tg) == 0:
            logger.warn("No POS lists for the language {}".format(lang_tg))

    def get_features(self, context_obj):
        if len(self.content) == 0:
            return []

        content_src, content_tg, verbs_src, verbs_tg, nouns_src, nouns_tg = 0, 0, 0, 0, 0, 0
        source_idx = context_obj['source_index']
        target_idx = context_obj['target_index']
        # check if source words are nouns, verbs, content words
        for word in context_obj['source_pos'][source_idx[0]:source_idx[1]]:
            content = False
            if self.belongs_to(word, self.nouns_src):
                nouns_src += 1
                if not content:
                    content_src += 1
                    content = True
            if self.belongs_to(word, self.verbs_src):
                verbs_src += 1
                if not content:
                    content_src += 1
                    content = True
            if not content:
                if self.belongs_to(word, self.content_src):
                    content_src += 1
        # check if target words are nouns, verbs, content words
        for word in context_obj['target_pos'][target_idx[0]:target_idx[1]]:
            content = False
            if self.belongs_to(word, self.nouns_tg):
                nouns_tg += 1
                if not content:
                    content_tg += 1
                    content = True
            if self.belongs_to(word, self.verbs_tg):
                verbs_tg += 1
                if not content:
                    content_tg += 1
                    content = True
            if not content:
                if self.belongs_to(word, self.content_tg):
                    content_tg += 1
        return [content_src/len(context_obj['source_token']),
                content_tg/len(context_obj['token']),
                verbs_src/len(context_obj['source_token']),
                verbs_tg/len(context_obj['source_token']),
                nouns_src/len(context_obj['source_token']),
                nouns_tg/len(context_obj['source_token'])]

    def get_feature_names(self):
        return ['percentage_content_words_src',
                'percentage_content_words_tg',
                'percentage_verbs_src',
                'percentage_verbs_tg',
                'percentage_nouns_src',
                'percentage_nouns_tg']
