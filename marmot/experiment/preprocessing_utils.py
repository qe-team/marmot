from __future__ import print_function

import os
import copy
import multiprocessing as multi
import logging
import numpy as np
from collections import defaultdict
from sklearn.preprocessing.label import LabelBinarizer, MultiLabelBinarizer
import ipdb

from marmot.util.simple_corpus import SimpleCorpus
from marmot.experiment.import_utils import list_of_lists

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('testlogger')


# convert alignments from list of strings 'i-j'
# to list of lists such that new_align[j] = i
def convert_alignments(align_list, n_words):
    new_align = [[] for i in range(n_words)]
    for pair in align_list:
        two_digits = pair.split('-')
        new_align[int(two_digits[1])].append(int(two_digits[0]))
    return new_align


# TODO: this function adds keys to the context object, but maybe the user wants different keys
# TODO: the function should be agnostic about which keys it adds -- why does it care?
# there is a difference between 'sequence fields' and 'token fields'
# this method creates a context for each token
def create_context(repr_dict, sentence_id=None):
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
#    if 'alignments' in repr_dict:
#        repr_dict['alignments'] = convert_alignments(repr_dict['alignments'], len(repr_dict['target']))

    active_keys = repr_dict.keys()
    active_keys.remove('tags')
    for idx, word in enumerate(repr_dict['target']):
        c = {}
        c['token'] = word
        c['index'] = idx
        if sentence_id is not None:
            c['sentence_id'] = sentence_id
        if type(repr_dict['tags']) == list or type(repr_dict['tags']) == np.ndarray:
            c['tag'] = repr_dict['tags'][idx]
            c['sequence_tags'] = repr_dict['tags']
        elif type(repr_dict['tags']) == int:
            c['tag'] = repr_dict['tags'][idx]
        else:
            print("Unknown type of tags representation:", type(repr_dict['tags']))
            return []
        for k in active_keys:
            c[k] = repr_dict[k]
        context_list.append(c)
    return context_list


# create context objects from a data_obj -
#     - a dictionary with representation labels as keys ('target', 'source', etc.) and
#       representations (lists of lists) as values
# output: if data_type = 'plain', one list of context objects is returned
#         if data_type = 'sequential', a list of lists of context objects is returned (list of sequences)
#         if data_type = 'token', a dict {token: <list_of_contexts>} is returned
# TODO: this function requires the 'target' and 'tag' keys, but the user may wish to specify other keys
# TODO: 'target' and 'tag' don't make sense for every task
def create_contexts(data_obj, data_type='plain'):
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

    for s_idx, sents in enumerate(zip(*data_obj.values())):
        if data_type == 'sequential':
            contexts.append(create_context({data_obj.keys()[i]: sents[i] for i in range(len(sents))}, sentence_id=s_idx))
        else:
            contexts.extend(create_context({data_obj.keys()[i]: sents[i] for i in range(len(sents))}, sentence_id=s_idx))

            # TODO: there is an error here
            if data_type == 'token':
                new_contexts = defaultdict(list)
                for cont in contexts:
                    new_contexts[cont['token']].append(cont)
                    contexts = copy.deepcopy(new_contexts)

    return contexts


# convert list of lists into a flat list
# TODO: there is an error where no code runs here
def flatten(lofl):
    if list_of_lists(lofl):
        return [item for sublist in lofl for item in sublist]
    elif type(lofl) == dict:
        return lofl.values()


def map_feature_extractor((context, extractor)):
    return extractor.get_features(context)


# feature extraction for categorical features with conversion to one-hot representation
# this implementation is for a list representation
# this returns a list of lists, where each list contains the feature extractor results for a context
# the point of returning a list of lists is to allow binarization of the feature values
# TODO: we can binarize over the columns of the matrix instead of binarizing the results of each feature extractor
# TODO: is the output of the single worker and the multithreaded different? if so, change
def contexts_to_features(contexts, feature_extractors, workers=1):
    # single thread
    if workers == 1:
        return [[x for a_list in [map_feature_extractor((context, extractor)) for extractor in feature_extractors] for x in a_list] for context in contexts]

    # multiple threads
    else:
        # resulting object
        res_list = []
        pool = multi.Pool(workers)
        logger.info('Multithreaded - Extracting the features for: ' + str(len(contexts)) + ' contexts...')
        # each context is paired with all feature extractors
        for extractor in feature_extractors:
            context_list = [(cont, extractor) for cont in contexts]
            features = pool.map(map_feature_extractor, context_list)
            res_list.append(features)
        # np.hstack and np.vstack can't be used because lists have objects of different types
        intermediate = [[x[i] for x in res_list] for i in range(len(res_list[0]))]
        res_list = [flatten(x) for x in intermediate]
        pool.close()
        pool.join()

    return res_list


# extract tags from a list of contexts
def tags_from_contexts(contexts):
    return [context['tag'] for context in contexts]


# train converters(binarizers) from categorical values to one-hot representation
#      for all features
# all_values is a list of lists, because we need to look at the feature values for every instance to binarize properly
def fit_binarizers(all_values):
    binarizers = {}
    for f in range(len(all_values[0])):
        cur_features = [context[f] for context in all_values]
        # only categorical values need to be binarized, ints/floats are left as they are
        if type(cur_features[0]) == str or type(cur_features[0]) == unicode:
            lb = LabelBinarizer()
            lb.fit(cur_features)
            binarizers[f] = lb
        elif type(cur_features[0]) == list:
            mlb = MultiLabelBinarizer()
            # default feature for unknown values
            cur_features.append(tuple(("__unk__",)))
            mlb.fit([tuple(x) for x in cur_features])
            binarizers[f] = mlb
    return binarizers


# convert categorical features to one-hot representations with pre-fitted binarizers
# TODO: this function implicitly converts the data into a numpy array
def binarize(features, binarizers):
    assert(list_of_lists(features))
    num_features = len(features[0])
    if binarizers != {} and max(binarizers.keys()) >= num_features:     
        print("Binarizers keys max: ", max(binarizers.keys()))
        print("Total feature number: ", num_features)
        print("Features:", features[0])
    assert(binarizers == {} or max(binarizers.keys()) < num_features)

    binarized_cols = []
    for i in range(num_features):
        # get this column
        cur_values = [f[i] for f in features]
        # if there's a binarizer for this column
        if i in binarizers:
            binarizer = binarizers[i]
            if type(binarizer) == LabelBinarizer:
                binarized_cols.append(binarizer.transform(cur_values))
            elif type(binarizer) == MultiLabelBinarizer:
                assert(list_of_lists(cur_values))
                # MultiLabelBinarizer doesn't support unknown values -- they need to be replaced with a default value
                # we're going to use the empty list as the default value
                cur_values_default = []
                default_value = binarizer.classes_[-1]
                for a_list in cur_values:
                    new_list = list(a_list)
                    for j, val in enumerate(new_list):
                        if val not in binarizer.classes_:
                            new_list[j] = default_value
                    cur_values_default.append(tuple(new_list))

                transformed = binarizer.transform(cur_values_default)
                binarized_cols.append(transformed)
            else:
                raise NotImplementedError('this function is not implemented for type: {}'.format(type(binarizer)))
        else:
            binarized_cols.append(np.array(cur_values).reshape(len(cur_values), 1))

    assert (len(binarized_cols) == num_features), 'the number of columns after binarization must match the number of features'
    new_features = np.hstack(binarized_cols)

    return new_features
