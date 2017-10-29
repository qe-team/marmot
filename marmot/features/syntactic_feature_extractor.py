from marmot.features.feature_extractor import FeatureExtractor
import sys


# get word's parent
def get_parent(dependencies, word_id, sentence_length, context_obj):
    for head in dependencies:
        if len(dependencies[head]) == 0:
            continue
        for dep in dependencies[head]:
            if dep['id'] == word_id:
                return head, dep['type']
#    assert(head < sentence_length), "Length: {}, head: {} for word {}, context object: {}".format(sentence_length, head, word_id, context_obj)
    return None, None


def get_siblings(dependencies, word_id, head_id, sentence_length):
    left_sib = 0
    right_sib = sentence_length
    if head_id is None:
        return None, None
    if head_id not in dependencies:
        return None, None
    for dep in dependencies[head_id]:
        if dep['id'] > word_id and dep['id'] < left_sib:
            left_sib = dep['id']
        if dep['id'] < word_id and dep['id'] > right_sib:
            right_sib = dep['id']
    if right_sib == sentence_length:
        right_sib = None
    if left_sib == 0:
        left_sib = None
    return left_sib, right_sib


class SyntacticFeatureExtractor(FeatureExtractor):

    def get_features_one_lang(self, dependencies, sentence, sentence_pos, word_idx, context_obj):
        sent_len = len(sentence)
        head, dep_type = get_parent(dependencies, word_idx, sent_len, context_obj)
        grandhead, grand_dep_type = get_parent(dependencies, head, sent_len, context_obj)
        left_sib, right_sib = get_siblings(dependencies, word_idx, head, sent_len)

#        try:
#            head_token = sentence[head] if head is not None else 'None'
#        except:
#            print('Head: {}, sentence: {}'.format(head, sentence))
#            sys.exit()
        if head == -1 and dep_type == 'root':
            head_token = 'None'
            head_pos = 'None'
        else:
            head_token = sentence[head] if head is not None else 'None'
            head_pos = sentence_pos[head] if head is not None else 'None'
        if grandhead == -1 and grand_dep_type == 'root':
            grand_head_token = 'None'
            grand_head_pos = 'None'
        else:
            grand_head_token = sentence[grandhead] if grandhead is not None else 'None'
            grand_head_pos = sentence_pos[grandhead] if grandhead is not None else 'None'
        left_sib_token = sentence[left_sib] if left_sib is not None else 'None'
        left_sib_pos = sentence_pos[left_sib] if left_sib is not None else 'None'
        right_sib_token = sentence[right_sib] if right_sib is not None else 'None'
        right_sib_pos = sentence_pos[right_sib] if right_sib is not None else 'None'
        token = sentence[word_idx]
        token_pos = sentence_pos[word_idx]
        synt_features = [str(dep_type), token + '|' + str(dep_type)]
        synt_features.extend([head_token + '|' + token, head_pos + '|' + token_pos])
        synt_features.extend([left_sib_token + '|' + token, right_sib_token + '|' + token, left_sib_pos + '|' + token_pos, right_sib_pos + '|' + token_pos])
        synt_features.extend([grand_head_token + '|' + head_token + '|' + token, grand_head_pos + '|' + head_pos + '|' + token_pos])
        return synt_features

    def get_features(self, context_obj):
        index = context_obj['index']
        src_index = context_obj['alignments'][index]
        if len(context_obj['source']) != len(context_obj['source_synt_pos']) or src_index is None:
            synt_features_src = ['None' for i in range(10)]
        else:
            synt_features_src = self.get_features_one_lang(context_obj['source_dependencies'], context_obj['source'], context_obj['source_pos'], src_index, context_obj)
        if len(context_obj['target']) != len(context_obj['target_synt_pos']):
            synt_features_tg = ['None' for i in range(10)]
        else:
            synt_features_tg = self.get_features_one_lang(context_obj['target_dependencies'], context_obj['target'], context_obj['target_pos'], index, context_obj)
        return synt_features_tg + synt_features_src


    def get_feature_names(self):
        return ['dep_type',
                'token+dep_type',
                'head+token',
                'head_pos+token_pos',
                'left_sib+token',
                'right_sib+token',
                'left_sib_pos+token_pos',
                'right_sib_pos+token_pos',
                'grandhead+head+token',
                'grandhead_pos+head_pos+token_pos',
                'src_dep_type',
                'src_token+dep_type',
                'src_head+token',
                'src_head_pos+token_pos',
                'src_left_sib+token',
                'src_right_sib+token',
                'src_left_sib_pos+token_pos',
                'src_right_sib_pos+token_pos',
                'src_grandhead+head+token',
                'src_grandhead_pos+head_pos+token_pos']
