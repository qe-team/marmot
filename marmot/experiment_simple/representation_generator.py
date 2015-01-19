import marmot
from marmot.experiment.experiment_utils import import_function

# generate an additional representation
class RepresentationGenerator():
    
    # <function> - function used to extract representations
    # <data> - list of representations
    def __init__(self, function, data, *args):
        self.func = import_function(function)
        self.args = args
        self.data = data if type(data) == list else [data]
        print "GENERATOR DATA", self.data

    # returns a pair (representation_label, representation_file), 
    # representation file should be sentence-aligned with target words
    def generate(self, data_obj):
        all_args = [ data_obj[d] for d in self.data ]
        all_args.extend(self.args)
        return self.func( *all_args )
