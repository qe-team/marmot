import os

import marmot
from marmot.experiment.experiment_utils import *
from marmot.util.simple_corpus import SimpleCorpus

# load and build object - universal
def build_object(obj_info, root_element='module'):
    print "IMPORT", obj_info[root_element]
    klass = import_class(obj_info[root_element])
    input_args = obj_info['args'] if 'args' in obj_info else []

    # map args to function outputs where requested
    for idx, arg in enumerate(input_args):
        if type(arg) is dict and 'type' in arg and arg['type'] == 'function_output':
            func = import_function(arg['func'])
            input_args[idx] = function_tree(func, arg['args'])

    # init the object
    obj = klass(*input_args)
    return obj 


def build_objects(object_list, root_element='module'):
    objects = []
    for obj_info in object_list:
        obj = build_object(obj_info)
        objects.append(obj)
    return objects

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
def create_contexts( data_obj, sequences=False ):
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
        if sequences:
            contexts.append( create_context( { data_obj.keys()[i]:sents[i] for i in range(len(sents)) } ) )
        else:
            contexts.extend( create_context( { data_obj.keys()[i]:sents[i] for i in range(len(sents)) } ) )

    return contexts
