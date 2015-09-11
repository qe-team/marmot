from __future__ import print_function, division

import sys
import numpy as np
from collections import Counter

###########################################################################
#
# This file contains different functions for generation of non-standard
# contexts (contexts where each 'token' is a list of words)
#
###########################################################################


# return the window of a list
# add symbols '_START_' and '_END_' if the range exceeds the length of the list
def negative_window(my_list, start, end):
    res = []
    while start < 0:
        res.append('_START_')
        start += 1
    while start < min(end, len(my_list)):
        res.append(my_list[start])
        start += 1
    while end > len(my_list):
        res.append('_END_')
        end -= 1
    return res


def create_context_ngram(repr_dict, order, bad_tagging="pessimistic"):
    '''
    :param repr_dict: a dict representing a 'line' or 'sentence' or a 'segment'
    :return: a list of context objects representing the data for each token in the sequence
    '''

    context_list = []
    # is checked before in create_contexts, but who knows
    if 'target' not in repr_dict:
        print("No 'target' label in data representations")
        return []
    if 'tags' not in repr_dict:
        print("No 'tag' label in data representations or wrong format of tag")
        print(repr_dict)
        return []

    active_keys = repr_dict.keys()
    active_keys.remove('tags')
    tag_map = {'OK': 1, 'BAD': 0}
    # if the order is greater than 1, we need to have the equal number of ngrams for each word
    # so we need to go beyond the borders of a sentence:
    # "this is my younger brother" has 3 3-grams: "this is my", "is my younger" and "my younger brother"
    # "my" participates in 3 trigrams, other words in 2 or less.
    # but we need all words to participate in 3 3-grams, so we create the following trigrams:
    # "_START_ _START_ this", "_START_ this is", "this is my", "is my younger" and "my younger brother", "younger brother _END_", "brother _END_ _END_"
    #logger.info("Order: {}".format(order))
    for i in range(len(repr_dict['target']) + order - 1):
        #logger.info("Word {}".format(i))
        c = {}
        #logger.info("Negative window from {} to {}, length {}".format(i - order + 1, i + 1, len(repr_dict['target'])))
        c['token'] = negative_window(repr_dict['target'], i - order + 1, i + 1)
        c['index'] = [i - order + 1, i + 1]
        # we take only tags for the existing tags
        # i.e. for the sequence "_START_ _START_ it" the tag will be the tag for "it" only
        tags = [tag_map[t] for t in repr_dict['tags'][max(0, i-order+1):min(len(repr_dict['tags']), i+1)]]
        c['tag'] = np.average(tags)
        for k in active_keys:
            c[k] = repr_dict[k]
        context_list.append(c)
        return context_list


# we don't really need the order here, it should always be None
# or anything else
# :test: -- True if data is test data, False if training -- test sentences can have empty source-segmentation field (if Moses failed to produce constrained reference for them)
# :only_target: -- True if only target sentence is segmented, needs to be processed without source segmentation
# :bad_tagging: -- tag all phrases with at least one bad word as "BAD"
#              if seg to False - only phrases with 50% or more bad words are tagged as "BAD"
def create_context_phrase(repr_dict, order=None, bad_tagging="pessimistic"):
    '''
    :param repr_dict: a dict representing a 'line' or 'sentence' or a 'segment'
    :return: a list of context objects representing the data for each token in the sequence
    '''

    context_list = []
    # is checked before in create_contexts, but who knows
    if 'target' not in repr_dict:
        print("No 'target' label in data representations")
        return []
    if 'tags' not in repr_dict:
        print("No 'tag' label in data representations or wrong format of tag")
        print(repr_dict)
        return []
    if 'segmentation' not in repr_dict or len(repr_dict['segmentation']) == 0:
        print("No 'segmentation' label in data representations")
        return []

    # no source segmentation means that no Moses segmentation was produced
    # in the training data we leave these sentences out
    # in the test data they are processed as normal
    # assuming that every target word is a separate segment
#    if ('source_segmentation' not in repr_dict or len(repr_dict['source_segmentation']) == 0) and not test and not only_target:
#        return []
    active_keys = repr_dict.keys()
    active_keys.remove('tags')
    if 'source_segmentation' in repr_dict:
        active_keys.remove('source_segmentation')
        if len(repr_dict['source_segmentation']) != 0 and len(repr_dict['source_segmentation']) != len(repr_dict['segmentation']):
            print("Wrong segmentation lengths: ", repr_dict)
            sys.exit()
    active_keys.remove('segmentation')
    for idx, (i, j) in enumerate(repr_dict['segmentation']):
#        print("Segment #{}: ({}, {})".format(idx, i, j))
        no_alignments = False
        c = {}
        c['token'] = repr_dict['target'][i:j]
        c['index'] = [i, j]
        # source phrase from the phrase segmentation
        if 'source_segmentation' in repr_dict and len(repr_dict['source_segmentation']) != 0:
            src_seg = repr_dict['source_segmentation'][idx]
            c['source_token'] = repr_dict['source'][src_seg[0]:src_seg[1]]
            c['source_index'] = (src_seg[0], src_seg[1])
        # source phrase from the alignments
        elif 'alignments' in repr_dict:
            alignments = []
            for ii in range(c['index'][0], c['index'][1]):
                alignments.extend(repr_dict['alignments'][ii])
            # converted to set to remove duplicates
            # converted back to list because set doesn't support indexing
            alignments = list(set(alignments))
            if len(alignments) == 0:
                no_alignments = True
                c['source_token'] = []
                c['source_index'] = []
            # source phrase -- substring between the 1st and the last word aligned to the target phrase
            # (unaligned words in between are included)
            else:
                c['source_token'] = [repr_dict['source'][ii] for ii in alignments]
                c['source_index'] = (alignments[0], alignments[-1] + 1)

        if len(c['token']) == 0:
            print("No token: from {} to {} in target: ".format(i, j), repr_dict['target'], repr_dict['source'], repr_dict['segmentation'])
        if j == 0:
            print("j==0!")
            print("Target: '{}', segmentation: {}, {}".format(' '.join(repr_dict['target']), i, j))
        if i == j or len(repr_dict['tags'][i:j]) == 0 or len(repr_dict['target'][i:j]) == 0:
            print("i==j!")
            print("Target: '{}', tags: '{}' segmentation: {}, {}".format(' '.join([w.encode('utf-8') for w in repr_dict['target']]), ' '.join(repr_dict['tags']), i, j))
        tags_cnt = Counter(repr_dict['tags'][i:j])

        # super-pessimistic tagging -- if BAD occurs any number of times - the final tag is BAD
        if bad_tagging == "super_pessimistic":
            if tags_cnt['BAD'] > 0:
                c['tag'] = 'BAD'
            else:
                c['tag'] = 'OK'
        # pessimistic tagging -- if BAD occurs in 1 of 3 words or more often -- the final tag is BAD
        elif bad_tagging == "pessimistic":
            if tags_cnt['BAD']/len(repr_dict['tags'][i:j]) < 0.3:
                c['tag'] = 'OK'
            else:
                c['tag'] = 'BAD'
        # optimisic - if OK occurs as much or more than BAD - the final tag is OK
        elif bad_tagging == "optimistic":
            if tags_cnt['OK'] >= tags_cnt['BAD']:
                c['tag'] = 'OK'
            else:
                c['tag'] = 'BAD'
        else:
            print("Unknown tag assignment scheme: {}".format(bad_tagging))

        for k in active_keys:
            c[k] = repr_dict[k]
        if len(c['source_token']) == 0 and not no_alignments:
            print("ERROR! Alignments exist, but source token is empty")
            sys.exit()
        context_list.append(c)
    return context_list


# create contexts where 'token' is an ngram of arbitrary length
# data_type is always 'plain' (no 'sequential' or 'token' for now)
# :order: -- order of ngram
# :data_type: -- 'plain' - data is a flat list
#                'sequential' - data is a list of sequences (used for dev and test)
def create_contexts_ngram(data_obj, order=None, data_type='plain', bad_tagging="pessimistic"):
    '''
    :param data_obj: an object representing a dataset consisting of files
    :param data_type:
    :return:
    '''
    contexts = []
    if 'target' not in data_obj:
        print("No 'target' label in data representations")
        return []

    if 'tags' not in data_obj:
        print("No 'tag' label in data representations or wrong format of tag")
        return []

    if 'segmentation' in data_obj:
        context_generator = create_context_phrase
    else:
        if order is None:
            print("The order of ngrams has to be defined to create the ngram contexts")
            return []
        context_generator = create_context_ngram

    print("Sentences in the data: {}".format(len(data_obj['target'])))
    if 'target_file' in data_obj:
        data_obj.pop('target_file')
    if 'source_file' in data_obj:
        data_obj.pop('source_file')
    overall = 0
    if data_type == 'plain':
        for s_idx, sents in enumerate(zip(*data_obj.values())):
            cont = context_generator({data_obj.keys()[i]: sents[i] for i in range(len(sents))}, order, bad_tagging=bad_tagging)
            overall += len(cont)
            contexts.extend(cont)
            print("Contexts: {}".format(overall))
    elif data_type == 'sequential':
        for s_idx, sents in enumerate(zip(*data_obj.values())):
            contexts.append(context_generator({data_obj.keys()[i]: sents[i] for i in range(len(sents))}, order, bad_tagging=bad_tagging))

    return contexts


# output a flat list of numbers
# a number for each context -- means the number of words this context represents
def get_contexts_words_number(contexts):
    numbers_list = []
    for c in contexts:
        if type(c['token']) is not list and type(c['token']) is not np.ndarray:
            numbers_list.append(1)
        else:
            numbers_list.append(len(c['token']))
    return numbers_list
