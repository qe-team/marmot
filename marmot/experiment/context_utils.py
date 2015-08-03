from __future__ import print_function

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


def create_context_ngram(repr_dict, order):
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
def create_context_phrase(repr_dict, order=None):
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
    if 'segmentation' not in repr_dict:
        print("No 'segmentation' label in data representations")
        return []

    active_keys = repr_dict.keys()
    active_keys.remove('tags')
    active_keys.remove('segmentation')
    tag_map = {'OK': 1, 'BAD': 0}
    for i, j in repr_dict['segmentation']:
        c = {}
        c['token'] = repr_dict['target'][i:j]
        c['index'] = [i, j]
        if j == 0:
            print("j==0!")
            print("Target: '{}', segmentation: {}, {}".format(' '.join(repr_dict['target'], i, j)))
#        c['tag'] = repr_dict['tags'][i:j]
#        tags = [tag_map[t] for t in repr_dict['tags'][i:j]]
#        c['tag'] = np.average(tags)
        tags_cnt = Counter(repr_dict['tags'][i:j])
        
        # pessimistic tagging -- if BAD occurs more often or as much as OK -- the final tag is BAD
        if tags_cnt['OK'] > tags_cnt['BAD']:
            c['tag'] = 'OK'
        else:
            c['tag'] = 'BAD'
        # super-pessimistic tagging -- if BAD occurs any number of times - the final tag is BAD
#        if tags_cnt['BAD'] > 0:
#            c['tag'] = 'BAD'
#        else:
#            c['tag'] = 'OK'

        #c['tag'] = tags_cnt.keys()[np.argmax(tags_cnt.values())]
 
        for k in active_keys:
            c[k] = repr_dict[k]
        context_list.append(c)
    return context_list


# create contexts where 'token' is an ngram of arbitrary length
# data_type is always 'plain' (no 'sequential' or 'token' for now)
# :order: -- order of ngram
# :data_type: -- 'plain' - data is a flat list
#                'sequential' - data is a list of sequences (used for dev and test)
def create_contexts_ngram(data_obj, order=None, data_type='plain'):
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

    if data_type == 'plain':
        for s_idx, sents in enumerate(zip(*data_obj.values())):
            contexts.extend(context_generator({data_obj.keys()[i]: sents[i] for i in range(len(sents))}, order))
#            print(contexts)
    elif data_type == 'sequential':
        for s_idx, sents in enumerate(zip(*data_obj.values())):
            contexts.append(context_generator({data_obj.keys()[i]: sents[i] for i in range(len(sents))}, order))

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
