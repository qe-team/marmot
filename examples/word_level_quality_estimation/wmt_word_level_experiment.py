from argparse import ArgumentParser
import yaml
import os, sys
import logging

import numpy as np

import marmot

from marmot.experiment import learning_utils
import marmot.experiment.experiment_utils as experiment_utils

from marmot.evaluation.evaluation_metrics import weighted_fmeasure

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('testlogger')

# define custom tag handler to join paths with the path of the word_level module
def join_with_module_path(loader, node):
    module_path = os.path.dirname(marmot.__file__)
    resolved = loader.construct_scalar(node)
    return os.path.join(module_path, resolved)

## register the tag handler
yaml.add_constructor('!join', join_with_module_path)

def main(config):
    # load ContextCreators from config file, run their input functions, and pass the result into the initialization function
    # init() all context creators specified by the user with their arguments
    # import them according to their fully-specified class names in the config file
    # it's up to the user to specify context creators which extract both negative and positive examples (if that's what they want)

    # Chris - working - we want to hit every token
    interesting_tokens = experiment_utils.import_and_call_function(config['interesting_tokens'])
    print "INTERESTING TOKENS: ", interesting_tokens
    logger.info('The number of interesting tokens is: ' + str(len(interesting_tokens)))
    workers = config['workers']

    # Note: context creators currently create their own interesting tokens internally (interesting tokens controls the index of the context creator)
    logger.info('building the context creators...')
    train_context_creators = experiment_utils.build_objects(config['context_creators'])

    # get the contexts for all of our interesting words (may be +,- or, multi-class)
    logger.info('mapping the training contexts over the interesting tokens in train...')
    train_contexts = experiment_utils.map_contexts(interesting_tokens, train_context_creators, workers=workers)

    # load and parse the test data
    logger.info('mapping the training contexts over the interesting tokens in test...')
    test_context_creator = experiment_utils.build_objects(config['testing'])
    test_contexts = experiment_utils.map_contexts(interesting_tokens, [test_context_creator])

    min_total = config['filters']['min_total']
    # filter token contexts based on the user-specified filter criteria
    logger.info('filtering the contexts by the total number of available instances...')
    train_contexts = experiment_utils.filter_contexts(train_contexts, min_total=min_total)
    test_contexts = experiment_utils.filter_contexts(test_contexts, min_total=min_total)

    # make sure the test_context and train_context keys are in sync
    experiment_utils.sync_keys(train_contexts, test_contexts)

    # test_contexts = filter_contexts(test_contexts, min_total=min_total)
    assert set(test_contexts.keys()) == set(train_contexts.keys())

    # extract the 'tag' attribute into the y-value for classification
    # tags may need to be converted to be consistent with the training data
    wmt_binary_classes = {u'BAD': 0, u'OK': 1}
    train_context_tags = experiment_utils.tags_from_contexts(train_contexts)
    train_context_tags = {k: np.array([wmt_binary_classes[v] for v in val]) for k, val in train_context_tags.items()}

    test_contexts = experiment_utils.convert_tagset(wmt_binary_classes, test_contexts)
    test_tags_actual = experiment_utils.tags_from_contexts(test_contexts)

    # all of the feature extraction should be parallelizable
    # note that a feature extractor MUST be able to parse the context exchange format, or it should throw an error:
    # { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
    feature_extractors = experiment_utils.build_feature_extractors(config['feature_extractors'])
    logger.info('mapping the feature extractors over the contexts for test...')
    test_context_features = experiment_utils.token_contexts_to_features_categorical(test_contexts, feature_extractors, workers=workers)
    logger.info('mapping the feature extractors over the contexts for train...')
    train_context_features = experiment_utils.token_contexts_to_features_categorical(train_contexts, feature_extractors, workers=workers)

    # flatten so that we can properly binarize the features
    all_values = experiment_utils.flatten(test_context_features.values())
    all_values.extend(experiment_utils.flatten(train_context_features.values()))
    binarizers = experiment_utils.fit_binarizers(all_values)
    test_context_features = {k: [experiment_utils.binarize(v, binarizers) for v in val] for k, val in test_context_features.items()}
    train_context_features = {k: [experiment_utils.binarize(v, binarizers) for v in val] for k, val in train_context_features.items()}

    # BEGIN LEARNING
    classifier_type = experiment_utils.import_class(config['learning']['classifier']['module'])
    # train the classifier for each token
    classifier_map = learning_utils.token_classifiers(train_context_features, train_context_tags, classifier_type)

    # classify the test instances
    # TODO: output a file in WMT format
    # WORKING - dump the output in WMT format
    logger.info('classifying the test instances')
    test_predictions = {}
    for key, features in test_context_features.iteritems():
        try:
            classifier = classifier_map[key]
            predictions = classifier.predict(features)
            test_predictions[key] = predictions
        except KeyError as e:
            print(key + " - is NOT in the classifier map")
            raise

    #### put the rest of the code into a separate 'evaluate' function that reads the WMT files

    # create the performance report for each word in the test data that we had a classifier for
    # TODO: Working - evaluate based on the format
    f1_map = {}
    for token, predicted in test_predictions.iteritems():
        logger.info("Evaluating results for token = " + token)
        actual = test_tags_actual[token]
        print 'Actual: ', actual
        print 'Predicted: ', predicted
        logger.info("\ttotal instances: " + str(len(predicted)))
        f1_map[token] = weighted_fmeasure(actual, predicted)

    logger.info('Printing the map of f1 scores by token: ')
    print(f1_map)


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

