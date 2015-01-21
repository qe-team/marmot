from argparse import ArgumentParser
import yaml
import logging

import marmot
from marmot.experiment_simple import experiment_utils_simple
from marmot.experiment import experiment_utils
from marmot.util.sequential_context_creator import SequentialContextCreator
from marmot.util.corpus_context_creator import CorpusContextCreator

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('testlogger')

def main(config):
    workers = config['workers']

    # unify data representations
    # test_data and training_data - lists of dictionaries { target: target_file, source: source_file, tags: tags).
    # <tags> can be a file with tags or a single tag
    test_data = experiment_utils.import_and_call_function(config['test'][0])
    training_data = experiment_utils.import_and_call_function(config['training'][0])
#    test_data = [ call_function(config['test']['func'], config['test']['args']) ]
#    train_data = [ call_function(config['training']['func'], config['training']['args']) ]


    # build objects for additional representations
    representation_generators = experiment_utils_simple.build_objects(config['additional'])

    # get additional representations
    for r in representation_generators:
        new_repr_test = r.generate(test_data)
        test_data[new_repr_test[0]] = new_repr_test[1]
        new_repr_train = r.generate(training_data)
        training_data[new_repr_train[0]] = new_repr_train[1]
#        new_repr_test = [ r.generate(tst) for tst in test_data ]
#        for idx, tst in enumerate(test_data):
#            tst[ new_repr_test[idx][0] ] = new_repr_test[idx][1]
#        new_repr_train = [ r.generate(train) for train in train_data ]
#        for idx, train in enumerate(train_data):
#            train[ new_repr_train[idx][0] ] = new_repr_train[idx][1]

    # since there is only one context creator and it does nothing, we don't need it any more
    # how to generate the old {token:context_list} representation?
    contexts_type = config['contexts'] if config.has_key('contexts') else 'plain'

    test_contexts = experiment_utils_simple.create_contexts(test_data, data_type=contexts_type)
    train_contexts = experiment_utils_simple.create_contexts(training_data, data_type=contexts_type)
 
    for r in representation_generators:
        r.cleanup()

#    print "TEST DATA", test_contexts
#    print "TRAINING DATA",train_contexts
 
    # make sure the test_context and train_context keys are in sync
#    experiment_utils.sync_keys(train_contexts, test_contexts)

    # test_contexts = filter_contexts(test_contexts, min_total=min_total)
#    assert set(test_contexts.keys()) == set(train_contexts.keys())

    # extract the 'tag' attribute into the y-value for classification
    # tags may need to be converted to be consistent with the training data
#    wmt_binary_classes = {u'BAD': 0, u'OK': 1}



    # extract contexts
    # filter contexts
    # extract features
    # binarise features

    # {token:contexts} format
    #if type(all_contexts) == dict:
    #    return {token:sequence_tags(contexts) for token, contexts in all_contexts.items()}

    #elif type(all_contexts) == list:
    #    # list of contexts
    #    if type(all_contexts[0]) == dict:
    #        return sequence_tags(all_contexts)
    #    # list of sequences of contexts
    #    elif type(all_contexts[0]) == list:
    #        return np.array([ sequence_tags(context) for context in all_contexts ])

    train_context_tags = experiment_utils_simple.call_for_each_element( train_contexts, experiment_utils_simple.tags_from_contexts, data_type=contexts_type )
    test_context_tags = experiment_utils_simple.call_for_each_element( test_contexts, experiment_utils_simple.tags_from_contexts, data_type=contexts_type )

#    if contexts_type == 'plain':
#        train_context_tags = experiment_utils_simple.tags_from_contexts(train_contexts)
#        test_context_tags = experiment_utils_simple.tags_from_contexts(test_contexts)
#    elif contexts_type == 'sequential':
#        pass
#    elif contexts_type == 'token':
#        pass


    #train_context_tags = experiment_utils_simple.tags_from_contexts(train_contexts)
    #test_context_tags = experiment_utils_simple.tags_from_contexts(test_contexts)

    print test_contexts[:10]
    print test_context_tags[:10]
    sys.exit(2)

    # all of the feature extraction should be parallelizable
    # note that a feature extractor MUST be able to parse the context exchange format, or it should throw an error:
    # { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
    feature_extractors = experiment_utils.build_feature_extractors(config['feature_extractors'])
    logger.info('mapping the feature extractors over the contexts for test...')
    test_context_features = experiment_utils_simple.contexts_to_features_categorical(test_contexts, feature_extractors, workers=workers)
    logger.info('mapping the feature extractors over the contexts for train...')
    train_context_features = experiment_utils_simple.contexts_to_features_categorical(train_contexts, feature_extractors, workers=workers)

    # flatten so that we can properly binarize the features
    all_values = experiment_utils.flatten(test_context_features)
    all_values.extend(experiment_utils.flatten(train_context_features))
    binarizers = experiment_utils.fit_binarizers(all_values)
    test_context_features = {k: [experiment_utils.binarize(v, binarizers) for v in val] for k, val in test_context_features.items()}
    train_context_features = {k: [experiment_utils.binarize(v, binarizers) for v in val] for k, val in train_context_features.items()}




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

