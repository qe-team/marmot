from __future__ import print_function

from argparse import ArgumentParser
import yaml
import logging
import copy

from marmot.experiment.import_utils import *
from marmot.experiment.preprocessing_utils import *
from marmot.experiment.learning_utils import map_classifiers, predict_all
from marmot.evaluation.evaluation_metrics import weighted_fmeasure
from marmot.evaluation.evaluation_utils import write_res_to_file

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('testlogger')


def main(config):
    workers = config['workers']
    print(config['test'][0]['output'])

    # unify data representations
    # test_data and training_data - lists of dictionaries { target: target_file, source: source_file, tags: tags).
    # <tags> can be a file with tags or a single tag
    test_data = import_and_call_function(config['test'][0])
    training_data = import_and_call_function(config['training'][0])

    # build objects for additional representations
    representation_generators = build_objects(config['additional'])

    # get additional representations
    # generators return a pair (label, representation)
    # TODO: generators should return data, not filenames
    # TODO: generators can check if the file they are trying to make already exists, if so, they should read it, if not, build the representation, and persist according to a flag
    for r in representation_generators:
        new_repr_test = r.generate(test_data)
        test_data[new_repr_test[0]] = new_repr_test[1]
        new_repr_train = r.generate(training_data)
        training_data[new_repr_train[0]] = new_repr_train[1]

    # since there is only one context creator and it does nothing, we don't need it any more
    # how to generate the old {token:context_list} representation?
    data_type = config['contexts'] if 'contexts' in config else 'plain'

    # TODO: `create_contexts` means 'read the files in the representation generator object'

    # TODO: files are implicitly whitespace tokenized
    # TODO: create_contexts maps a whitespace tokenized, line by line dataset into one of our three data representations
    test_contexts = create_contexts(test_data, data_type=data_type)
    train_contexts = create_contexts(training_data, data_type=data_type)

    print('TEST contexts', len(test_contexts))
    for r in representation_generators:
        r.cleanup()

    # make sure the test_context and train_context keys are in sync
    # TODO: this is important when we are learning token-level classifiers
#    experiment_utils.sync_keys(train_contexts, test_contexts)

    # TODO: this is important when we are learning token-level classifiers
    # test_contexts = filter_contexts(test_contexts, min_total=min_total)
#    assert set(test_contexts.keys()) == set(train_contexts.keys())

    train_tags = call_for_each_element(train_contexts, tags_from_contexts, data_type=data_type)
    test_tags = call_for_each_element(test_contexts, tags_from_contexts, data_type=data_type)
    print('TEST tags', len(test_tags))

    # all of the feature extraction should be parallelizable
    # note that a feature extractor MUST be able to parse the context exchange format, or it should throw an error:
    # { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
    feature_extractors = build_objects(config['feature_extractors'])
    logger.info('mapping the feature extractors over the contexts for test...')
    test_features = call_for_each_element(test_contexts, contexts_to_features, [feature_extractors, workers], data_type=data_type)
    logger.info('mapping the feature extractors over the contexts for train...')
    train_features = call_for_each_element(train_contexts, contexts_to_features, [feature_extractors, workers], data_type=data_type)
    print('TEST features', len(test_features))
    # flatten so that we can properly binarize the features
    all_values = []
    if data_type == 'sequential':
        all_values = flatten(train_features)
    elif data_type == 'plain':
        all_values = copy.deepcopy(train_features)
    elif data_type == 'token':
        all_values = flatten(train_features.values())
    logger.info('fitting binarizers...')
    binarizers = fit_binarizers(all_values)
    logger.info('binarizing test data...')
    test_features = call_for_each_element(test_features, binarize, [binarizers], data_type=data_type)
    logger.info('binarizing training data...')
    train_features = call_for_each_element(train_features, binarize, [binarizers], data_type=data_type)
    logger.info('training sets successfully generated')
    print('TEST features binary', len(test_features))
    # learning
    if data_type == 'sequential':
        pass
    else:
        logger.info('start training...')
        classifier_type = import_class(config['learning']['classifier']['module'])
        # train the classifier(s)
        classifier_map = map_classifiers(train_features, train_tags, classifier_type, data_type=data_type)

    logger.info('classifying the test instances')
    test_predictions = predict_all(test_features, classifier_map, data_type=data_type)
    print('TEST predictions', len(test_predictions))

    if data_type == 'token':
        f1_map = {}
        for token, predicted in test_predictions.iteritems():
            logger.info("Evaluating results for token = " + token)
            actual = test_tags_actual[token]
#            print('Actual: ', actual)
#            print('Predicted: ', predicted)
#            logger.info("\ttotal instances: " + str(len(predicted)))
            f1_map[token] = weighted_fmeasure(actual, predicted)
            logger.info('Printing the map of f1 scores by token: ')
        print(f1_map)
    elif data_type == 'plain':
        f1 = weighted_fmeasure(test_tags, test_predictions)
        logger.info('F1 score: ')
        print(f1)

    write_res_to_file(config['test'][0]['output'], test_predictions)


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

