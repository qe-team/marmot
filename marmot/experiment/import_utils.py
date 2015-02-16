def import_class(module_name):
    mod_name, class_name = module_name.rsplit('.', 1)
    mod = __import__(mod_name, fromlist=[class_name])
    klass = getattr(mod, class_name)
    return klass


def import_function(func_name):
    mod_name, func_name = func_name.rsplit('.', 1)
    mod = __import__(mod_name, fromlist=[func_name])
    func = getattr(mod, func_name)
    return func


def call_function(function, args):
    return function(*args)


def import_and_call_function(function_obj):
    func = import_function(function_obj['func'])
    args = function_obj['args']
    return call_function(func, args)


# check that <a_list> is a list of lists
def list_of_lists(a_list):
    if type(a_list) == list and len(a_list) > 0 and all([type(l) == list for l in a_list]):
        return True
    return False


# call the same function for the data organised in different structures
def call_for_each_element(data, function, args=[], data_type='sequential'):
    if data_type == 'plain':
        assert(not list_of_lists(data[0]))
        return function(data, *args)
    elif data_type == 'sequential':
        assert(list_of_lists(data[0]))
        return [function(d, *args) for d in data]
    elif data_type == 'token':
        assert(type(data) == dict)
    return {token: function(contexts, *args) for token, contexts in data.items()}


# call through the function tree, at each node look for: (func:<>, args:<>)
# args[] may have a property: type: function_output:, if so, call recursively with (func:<>, args:<>)
# finally, call the original func with its args
def function_tree(func, args):
    # map args to function outputs where requested
    for idx, arg in enumerate(args):
        if type(arg) is dict and 'type' in arg and arg['type'] == 'function_output':
            inner_func = import_function(arg['func'])
            args[idx] = function_tree(inner_func, arg['args'])

    # the function is ready to be called
    return call_function(func, args)


# load and build object - universal
def build_object(obj_info, root_element='module'):
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
