import os
import numpy as np
import multiprocessing as multi
import logging
import copy
from collections import defaultdict

import marmot
from marmot.experiment import experiment_utils
from marmot.util.simple_corpus import SimpleCorpus

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('testlogger')

# load and build object - universal
def build_object(obj_info, root_element='module'):
    print "IMPORT", obj_info[root_element]
    klass = experiment_utils.import_class(obj_info[root_element])
    input_args = obj_info['args'] if 'args' in obj_info else []

    # map args to function outputs where requested
    for idx, arg in enumerate(input_args):
        if type(arg) is dict and 'type' in arg and arg['type'] == 'function_output':
            func = experiment_utils.import_function(arg['func'])
            input_args[idx] = experiment_utils.function_tree(func, arg['args'])

    # init the object
    obj = klass(*input_args)
    return obj 


def build_objects(object_list, root_element='module'):
    objects = []
    for obj_info in object_list:
        obj = build_object(obj_info)
        objects.append(obj)
    return objects

# convert alignments from list of strings 'i-j' to list of lists such that new_align[j] = i
def convert_alignments(align_list, n_words):
    new_align = [ [] for i in range(n_words) ]
    for pair in align_list:
        two_digits = pair.split('-')
        new_align[int(two_digits[1])].append(int(two_digits[0]))
    return new_align
        

def create_context( repr_dict ):
    context_list = []
    # is checked before in create_contexts, but who knows
    if not repr_dict.has_key('target'):
        print "No 'target' label in data representations"
        return []
    if not repr_dict.has_key('tag') or not (type(repr_dict['tag']) == list or type(repr_dict['tag']) == int):
        print "No 'tag' label in data representations or wrong format of tag"
        return []
    if repr_dict.has_key('alignments'):
        repr_dict['alignments'] = convert_alignments(repr_dict['alignments'], len(repr_dict['target']))

    active_keys = repr_dict.keys()
    active_keys.remove('tag')
    for idx, word in enumerate(repr_dict['target']):
        c = {}
        c['token'] = word
        c['index'] = idx
        c['tag'] = repr_dict['tag'] if type(repr_dict['tag']) == int else repr_dict['tag'][idx]
        for k in active_keys:
            c[k] = repr_dict[k]
        context_list.append(c)
    return context_list


# create context objects from a data_obj: a dictionary with representation labels as keys ('target', 'source', etc.) and files as values
# output: if sequences = False, one list of context objects is returned
#         if sequences = True, list of lists of context objects is returned (list of sequences)
def create_contexts( data_obj, data_type='plain' ):
    contexts = []
    if not data_obj.has_key('target'):
        print "No 'target' label in data representations"
        return []

    if not data_obj.has_key('tag') or not (os.path.isfile(data_obj['tag']) or type(data_obj['tag']) == int):
        print "No 'tag' label in data representations or wrong format of tag"
        print data_obj
        return []

    corpora = [ SimpleCorpus(d) for d in data_obj.values() ]
    #print data_obj
    for sents in zip(*[c.get_texts_raw() for c in corpora]):
        if data_type == 'sequential':
            contexts.append( create_context( { data_obj.keys()[i]:sents[i] for i in range(len(sents)) } ) )
        else:
            contexts.extend( create_context( { data_obj.keys()[i]:sents[i] for i in range(len(sents)) } ) )
            if data_type == 'token':
                new_contexts = defaultdict(list)
                for cont in contexts:
                    new_contexts[cont['token']].append( cont )
                contexts = copy.deepcopy(new_contexts)

    return contexts

# extract tags from a list of contexts
def tags_from_contexts(contexts):
    return np.array([ context['tag'] for context in contexts ])


# returns a numpy array
def map_feature_extractors((context, extractor)):
#    return np.hstack([extractor.get_features(context) for extractor in feature_extractors])
    return extractor.get_features(context)


# feature extraction for categorical features with convertation to one-hot representation
def contexts_to_features_categorical(contexts, feature_extractors, workers=1):
    #single thread
    if workers == 1:
        return [[x for a_list in [map_feature_extractors((context, extractor)) for extractor in feature_extractors] for x in a_list ] for context in contexts]

    #multiple threads
    else:
        #resulting object
        res_list = []
        pool = multi.Pool(workers)
        print("Feature extractors: ", feature_extractors)
        logger.info('Multithreaded - Extracting categorical contexts -- ' + str(len(contexts)) + ' contexts...')
        print "CONTEXTS", type(contexts), type(contexts[0]), len(contexts), len(contexts[0])
        #each context is paired with all feature extractors
        for extractor in feature_extractors:
            context_list = [(cont, extractor) for cont in contexts]
            res_list.append( pool.map(map_feature_extractors, context_list) )
        # np.hstack and np.vstack can't be used because lists have objects of different types
        intermediate =  [ [x[i] for x in res_list] for i in range(len(res_list[0])) ]
        res_list = [ experiment_utils.flatten(sl) for sl in intermediate ]

    return res_list


# check that <a_list> is a list of lists
def list_of_lists(a_list):
    if type(a_list) == list and len(a_list) > 0 and type(a_list[0]) == list:
        return True
    return False


# call the same function for the data organised in different structures
def call_for_each_element(data, function, args=[], data_type='sequential'):
    if data_type == 'plain':
        assert( not list_of_lists(data) )
        return function(data, *args)
    elif data_type == 'sequential':
        assert( list_of_lists(data) )
        return [ function(d, *args) for d in data ]
    elif data_type == 'token':
        assert( type(data) == dict )
        return {token:function(contexts, *args) for token, contexts in data.items()}

