from __future__ import division
import numpy as np
import multiprocessing as multi
import logging
import types
import sklearn
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

from import_utils import import_class
from preprocessing_utils import map_feature_extractor

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('testlogger')


def init_class(klass, args):
    return klass(*args)


# def build_context_creator(creator_obj):
#     creator_klass = import_class(creator_obj['module'])
#     input_args = creator_obj['args']
#
#     # map args to function outputs where requested
#     for idx, arg in enumerate(input_args):
#         if type(arg) is dict and 'type' in arg and arg['type'] == 'function_output':
#             func = import_function(arg['func'])
#             input_args[idx] = function_tree(func, arg['args'])
#
#     # init the object
#     creator = creator_klass(*input_args)
#     return creator
#
#
# def build_context_creators(context_creator_list):
#     context_creators = []
#     for creator_obj in context_creator_list:
#         creator = build_context_creator(creator_obj)
#         context_creators.append(creator)
#     return context_creators


def filter_contexts(token_contexts, min_total=1):
    return {token: contexts for token, contexts in token_contexts.items() if len(contexts) >= min_total}


# filter contexts to satisfy the whole size constraint and the class size constraint
def filter_contexts_class(token_contexts, min_total=1, min_class_count=1, proportion=2):
    new_token_contexts = {}
    classes = set([cc['tag'] for context in token_contexts.values() for cc in context])
    for token, contexts in token_contexts.items():
        # no need to check other conditions if there are too few contexts
        if len(contexts) < min_total:
            continue

        class_counts = {cl: 0 for cl in classes}
        for cc in contexts:
            class_counts[cc['tag']] += 1
        min_class = min(class_counts.values())
        cur_proportion = max(class_counts.values())/max(min_class,1)

        if min_class >= min_class_count and cur_proportion <= proportion:
            new_token_contexts[token] = contexts

    return new_token_contexts


# convert the tag representation of a list of contexts into another format (remap the tag strings)
import copy
def convert_tagset(tagmap, tok_contexts):
    tok_contexts_copy = copy.deepcopy(tok_contexts)
    for tok, contexts in tok_contexts_copy.iteritems():
        for context in contexts:
            context['tag'] = tagmap[context['tag']]
    return tok_contexts_copy


def flatten(lofl):
    return [item for sublist in lofl for item in sublist]

# returns a dict of token --> contexts
# remember that contexts store their own tags
def map_contexts(tokens, context_creators):
    return {token: flatten([creator.get_contexts(token) for creator in context_creators]) for token in tokens}

def map_context_creators((token, context_creators)):
    logger.info('mapping context creators for token: ' + token)
    contexts = flatten([creator.get_contexts(token) for creator in context_creators])
    return token, contexts


#multithreaded context mapping
def map_contexts(tokens, context_creators, workers=1):
    #single thread
    if workers == 1:
        return {token: flatten([creator.get_contexts(token) for creator in context_creators]) for token in tokens}
    #multiple threads
    else:
        # res_dict = {}
        pool = multi.Pool(workers)
        tokens_with_extractors = [(token, context_creators) for token in tokens]
        res = pool.map(map_context_creators, tokens_with_extractors)
        res_dict = {k:v for k,v in res}
        return res_dict


# multithreaded feature extraction
# this is for a dict representation {<tpk>: [<context>,...]}
def token_contexts_to_features(token_contexts, feature_extractors, workers=1):
    #single thread
    if workers == 1:
         return {token: np.vstack( [np.hstack([map_feature_extractor((context, extractor)) for extractor in feature_extractors] ) for context in contexts]) for token, contexts in token_contexts.items()}

    #multiple threads
    else:
        #resulting object
        res_dict = {}
        pool = multi.Pool(workers)
        print("Feature extractors: ", feature_extractors)
        for token, contexts in token_contexts.items():
            logger.info('Multithreaded - Extracting contexts for token: ' + token + ' -- with ' + str(len(contexts)) + ' contexts...')
            #each context is paired with all feature extractors
#            context_list = [ (cont, feature_extractors) for cont in contexts ]
            extractors_output = []
            for extractor in feature_extractors:
                context_list = [(cont, extractor) for cont in contexts]
                extractors_output.append(np.vstack(pool.map(map_feature_extractor, context_list)))
            res_dict[token] = np.hstack(extractors_output)

        return res_dict


# feature extraction for categorical features with convertation to one-hot representation
# TODO: this is unused code right now -- remove or update in favor of the new 'data_type' param
def token_contexts_to_features_categorical(token_contexts, feature_extractors, workers=1):
    #single thread
    if workers == 1:
        return {token: [[x for a_list in [map_feature_extractor((context, extractor)) for extractor in feature_extractors] for x in a_list ] for context in contexts] for token, contexts in token_contexts.items()}

    #multiple threads
    else:
        #resulting object
        res_dict = {}
        pool = multi.Pool(workers)
        print("Feature extractors: ", feature_extractors)
        for token, contexts in token_contexts.items():
            logger.info('Multithreaded - Extracting categorical contexts for token: ' + token + ' -- with ' + str(len(contexts)) + ' contexts...')
            #each context is paired with all feature extractors
            extractors_output = []
            for extractor in feature_extractors:
                context_list = [(cont, extractor) for cont in contexts]
                extractors_output.append( pool.map(map_feature_extractor, context_list) )
            # np.hstack and np.vstack can't be used because lists have objects of different types
            intermediate =  [ [x[i] for x in extractors_output] for i in range(len(extractors_output[0])) ]
            res_dict[token] = [ flatten(sl) for sl in intermediate ]

        return res_dict


def feature_names_from_extractor_list(feature_extractors):
    """
    get a list of feature names from a list of feature extractors
    :param feature_extractors: a list of feature extractors
    :return: a list of the feature names for each extractor (think first row of .csv file)
    """
    feature_names = [feature_name for feature_extractor in feature_extractors for feature_name in feature_extractor.get_feature_names()]
    return feature_names


def tags_from_contexts(token_contexts):
    """
    create a dict mapping tokens to their tags
    :param token_contexts:
    :return: a dict of {<token>: [tag_i, tag_i+1, ...]}
    """
    return {token: np.array([context['tag'] for context in contexts]) for token, contexts in token_contexts.items()}


def sync_keys(dict_a, dict_b):
    '''
    make sure two dicts have the same keys, delete the ones that are different
    :param dict_a:
    :param dict_b:
    :return:
    '''
    dict_a_keys = set(dict_a.keys())
    dict_b_keys = set(dict_b.keys())
    for k in dict_a_keys.symmetric_difference(dict_b_keys):
        if k in dict_a_keys:
            del dict_a[k]
        else:
            del dict_b[k]
