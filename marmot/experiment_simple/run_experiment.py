from argparse import ArgumentParser
import yaml

import marmot
from marmot.experiment_simple.experiment_utils_simple import *
from marmot.experiment.experiment_utils import *
from marmot.util.corpus_context_creator import CorpusContextCreator

def main(config):

    # unify data representations
    # test_data and training_data - lists of dictionaries { target: target_file, source: source_file, tags: tags).
    # <tags> can be a file with tags or a single tag
    print "CONFIG TEST", config['test']#, config['test']['func'], config['test']['args']
    print "CONFIG TRAIN", config['training']#, config['training']['func'], config['training']['args']
    test_data = import_and_call_function(config['test'][0])
    training_data = import_and_call_function(config['training'][0])
#    test_data = [ call_function(config['test']['func'], config['test']['args']) ]
#    train_data = [ call_function(config['training']['func'], config['training']['args']) ]


    # build objects for additional representations
    representation_generators = build_objects(config['additional'])

    # get additional representations
    for r in representation_generators:
        new_repr_test = r.generate(test_data)
        print "NEW REPRESENTATION", new_repr_test
        test_data[new_repr_test[0]] = new_repr_test[1]
        new_repr_train = r.generate(training_data)
        training_data[new_repr_train[0]] = new_repr_train[1]
#        new_repr_test = [ r.generate(tst) for tst in test_data ]
#        for idx, tst in enumerate(test_data):
#            tst[ new_repr_test[idx][0] ] = new_repr_test[idx][1]
#        new_repr_train = [ r.generate(train) for train in train_data ]
#        for idx, train in enumerate(train_data):
#            train[ new_repr_train[idx][0] ] = new_repr_train[idx][1]

    print "TEST DATA:", test_data
    print "TRAINING DATA:",training_data
    all_contexts = create_contexts(test_data)
    print "ALL CONTEXTS", all_contexts
    test_contexts = CorpusContextCreator( create_contexts(test_data) )
    train_contexts = CorpusContextCreator( create_contexts(train_data) )
    print "TEST DATA:", test_contexts
    print "TRAINING DATA:",training_contexts
 

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("configuration_file", action="store", help="path to the config file (in YAML format).")
    args = parser.parse_args()
    config = {}

    # Experiment hyperparams
    cfg_path = args.configuration_file
    # read configuration file
    with open(cfg_path, "r") as cfg_file:
        config = yaml.load(cfg_file.read())
    main(config)

