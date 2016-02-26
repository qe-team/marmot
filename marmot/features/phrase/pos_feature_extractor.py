from __future__ import division

from collections import defaultdict
import logging

from marmot.features.feature_extractor import FeatureExtractor

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('testlogger')


# collections of content tags for some languages
def get_tags(lang):
    content, verbs, nouns, pronouns = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    content['english'] = ['JJ', 'RB', 'NN', 'VB']
    content['spanish'] = ['ADJ', 'ADV', 'NC', 'NMEA', 'NMON', 'NP', 'VL']
    content['german'] = ['ADJ', 'ADV', 'NN', 'NE', 'VA', 'VV', 'VM']
    verbs['english'] = ['VB']
    verbs['spanish'] = ['VL']
    verbs['german'] = ['VA', 'VV', 'VM']
    nouns['english'] = ['NN']
    nouns['spanish'] = ['NC', 'NMEA', 'NMON', 'NP']
    nouns['german'] = ['NN', 'NE']
    pronouns['english'] = ['PP', 'WP$']
    pronouns['spanish'] = ['DM', 'INT', 'PP', 'REL']
    pronouns['german'] = ['PPOS', 'PD', 'PI', 'PREL', 'PW', 'PAV']
    return content[lang], nouns[lang], verbs[lang], pronouns[lang]


class POSFeatureExtractor(FeatureExtractor):

    def belongs_to(self, word_tag, category):
        for tag in category:
            if word_tag.startswith(tag):
                return True
        return False

    def __init__(self, lang_src, lang_tg):
        self.content_src, self.nouns_src, self.verbs_src, self.pronouns_src = get_tags(lang_src)
        self.content_tg, self.nouns_tg, self.verbs_tg, self.pronouns_tg = get_tags(lang_tg)
        if len(self.content_src) == 0:
            logger.warn("No POS lists for the language {}".format(lang_src))
        if len(self.content_tg) == 0:
            logger.warn("No POS lists for the language {}".format(lang_tg))

    def get_features(self, context_obj):
        if len(self.content_src) == 0 or len(self.content_tg) == 0:
            logger.warn("One or more POS lists are empty")

        content_src, content_tg, verbs_src, verbs_tg, nouns_src, nouns_tg, pronouns_src, pronouns_tg = 0, 0, 0, 0, 0, 0, 0, 0
        source_idx = context_obj['source_index']
        target_idx = context_obj['index']
        # check if source words are nouns, verbs, content words
        if len(source_idx) > 0:
            for word in context_obj['source_pos'][source_idx[0]:source_idx[1]]:
                content = False
                if self.belongs_to(word, self.pronouns_src):
                    pronouns_src += 1
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
            if self.belongs_to(word, self.pronouns_tg):
                pronouns_tg += 1
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
        content_tg_percent = content_tg/len(context_obj['token'])
        verbs_tg_percent = verbs_tg/len(context_obj['token'])
        nouns_tg_percent = nouns_tg/len(context_obj['token'])
        pronouns_tg_percent = pronouns_tg/len(context_obj['token'])
        content_src_percent = 0
        verbs_src_percent = 0
        nouns_src_percent = 0
        pronouns_src_percent = 0
        if len(context_obj['source_token']) > 0:
            content_src_percent = content_src/len(context_obj['source_token'])
            verbs_src_percent = verbs_src/len(context_obj['source_token'])
            nouns_src_percent = nouns_src/len(context_obj['source_token'])
            pronouns_src_percent = pronouns_src/len(context_obj['source_token'])
        ratio_content = content_src_percent/content_tg_percent if content_tg_percent > 0 else (1 if content_src_percent == 0 else 0)
        ratio_verbs = verbs_src_percent/verbs_tg_percent if verbs_tg_percent > 0 else (1 if verbs_src_percent == 0 else 0)
        ratio_nouns = nouns_src_percent/nouns_tg_percent if nouns_tg_percent > 0 else (1 if nouns_src_percent == 0 else 0)
        ratio_pronouns = pronouns_src_percent/pronouns_tg_percent if pronouns_tg_percent > 0 else (1 if pronouns_src_percent == 0 else 0)

        return [content_src_percent,
                content_tg_percent,
                verbs_src_percent,
                verbs_tg_percent,
                nouns_src_percent,
                nouns_tg_percent,
                pronouns_src_percent,
                pronouns_tg_percent,
                ratio_content,
                ratio_verbs,
                ratio_nouns,
                ratio_pronouns]

    def get_feature_names(self):
        return ['percentage_content_words_src',
                'percentage_content_words_tg',
                'percentage_verbs_src',
                'percentage_verbs_tg',
                'percentage_nouns_src',
                'percentage_nouns_tg',
                'percentage_pronouns_src',
                'percentage_pronouns_tg',
                'ratio_content_words_src_tg',
                'ratio_verbs_src_tg',
                'ratio_nouns_src_tg',
                'ratio_pronouns_src_tg']
