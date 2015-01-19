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

def create_context( repr_dict ):
    context_list = []
    # is checked before in create_contexts, but who knows
    if not repr_dict.has_key('target'):
        print "No 'target' label in data representations"
        return []
    for idx, word in enumerate(repr_dict['target']):
        c = {}
        c['token'] = word
        c['index'] = idx
        for k in repr_dict.keys():
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
    corpora = [ SimpleCorpus(d) for d in data_obj.values() ]
    for sents in zip(*[c.get_texts() for c in corpora]):
        if sequences:
            contexts.append( create_context( { data_obj.keys()[i]:sents[i] for i in range(len(sents)) } ) )
        contexts.extend( create_context( { data_obj.keys()[i]:sents[i] for i in range(len(sents)) } ) )
