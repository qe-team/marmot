from __future__ import print_function, division

from argparse import ArgumentParser
import yaml
import logging
import copy

from marmot.experiment.import_utils import *
from marmot.experiment.preprocessing_utils import *
from marmot.experiment.learning_utils import map_classifiers, predict_all
from marmot.evaluation.evaluation_metrics import weighted_fmeasure
from marmot.evaluation.evaluation_utils import compare_vocabulary
from marmot.evaluation.evaluation_utils import write_res_to_file

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('experiment_logger')


def main(config):
    workers = config['workers']

    # REPRESENTATION GENERATION
    # build objects for additional representations
    representation_generators = build_objects(config['representations'])

    # get additional representations
    # generators return a pair (label, representation)
    train_data_generator = build_object(config['datasets']['training'][0])
    test_data_generator = build_object(config['datasets']['test'][0])
    train_data = train_data_generator.generate()
    test_data = test_data_generator.generate()
    for r in representation_generators:
        train_data = r.generate(train_data)
        test_data = r.generate(test_data)

    logger.info('here are the keys in your representations: {}'.format(train_data.keys()))

    # the data_type is the format corresponding to the model of the data that the user wishes to learn
    data_type = config['contexts'] if 'contexts' in config else 'plain'

    test_contexts = create_contexts(test_data, data_type=data_type)
    train_contexts = create_contexts(train_data, data_type=data_type)

    logger.info('Vocabulary comparison -- coverage for each dataset: ')
    logger.info(compare_vocabulary([train_data['target'], test_data['target']]))
 
    # END REPRESENTATION GENERATION

    # FEATURE EXTRACTION
    # make sure the test_context and train_context keys are in sync
    # TODO: this is important when we are learning token-level classifiers
#    experiment_utils.sync_keys(train_contexts, test_contexts)

    # TODO: this is important when we are learning token-level classifiers
    # test_contexts = filter_contexts(test_contexts, min_total=min_total)
#    assert set(test_contexts.keys()) == set(train_contexts.keys())

    train_tags = call_for_each_element(train_contexts, tags_from_contexts, data_type=data_type)
    test_tags = call_for_each_element(test_contexts, tags_from_contexts, data_type=data_type)

    # all of the feature extraction should be parallelizable
    # note that a feature extractor MUST be able to parse the context exchange format, or it should throw an error:
    # { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
    logger.info('creating feature extractors...')
    feature_extractors = build_objects(config['feature_extractors'])
    logger.info('mapping the feature extractors over the contexts for test...')
    test_features = call_for_each_element(test_contexts, contexts_to_features, [feature_extractors, workers], data_type=data_type)
    logger.info('mapping the feature extractors over the contexts for train...')
    train_features = call_for_each_element(train_contexts, contexts_to_features, [feature_extractors, workers], data_type=data_type)

    logger.info('number of training instances: {}'.format(len(train_features)))
    logger.info('number of testing instances: {}'.format(len(test_features)))

    logger.info('All of your features now exist in their raw representation, but they may not be numbers yet')
    # END FEATURE EXTRACTION

    # BEGIN CONVERTING FEATURES TO NUMBERS

    logger.info('binarization flag: {}'.format(config['features']['binarize']))
    # flatten so that we can properly binarize the features
    if config['features']['binarize'] is True:
        logger.info('Binarizing your features...')
        all_values = []
        if data_type == 'sequential':
            all_values = flatten(train_features)
        elif data_type == 'plain':
            all_values = train_features
        elif data_type == 'token':
            all_values = flatten(train_features.values())

        logger.info('fitting binarizers...')
        binarizers = fit_binarizers(all_values)
        logger.info('binarizing test data...')
        test_features = call_for_each_element(test_features, binarize, [binarizers], data_type=data_type)
        logger.info('binarizing training data...')
        # TODO: this line hangs with alignment+w2v
        train_features = call_for_each_element(train_features, binarize, [binarizers], data_type=data_type)

        logger.info('All of your features are now scalars in numpy arrays')

    logger.info('training and test sets successfully generated')

    # the way that we persist depends upon the structure of the data (plain/sequence/token_dict)
    # TODO: remove this once we have a list containing all datasets
    if config['features']['persist']:
        experiment_datasets = [{'name': 'test', 'features': test_features}, {'name': 'train', 'features': train_features}]
        # persist features to csv with the labels as the last column
        if config['features']['persist_dir']:
            persist_dir = config['features']['persist_dir']
        else:
            persist_dir = os.path.getcwd()
        logger.info('persisting your features to: '.format(persist_dir))
        # for each dataset, write a file and persist the features
        for dataset_obj in experiment_datasets:
            pass
            # save_features()



    # TODO: we should only learn and evaluate the model if this is what the user wants
    # TODO: we should be able to dump the features for each of the user's datasets to a file specified by the user

    # BEGIN LEARNING

    # TODO: different sequence learning modules need different representation, we should wrap them in a class
    # TODO: create a consistent interface to sequence learners, will need to use *args and **kwargs because APIs are very different
    import ipdb
    from sklearn.metrics import f1_score
    import numpy as np
    if data_type == 'sequential':
        logger.info('training sequential model...')

        # TODO: move the tag and array conversion code to the utils of this module
        # TODO: check if X and y are in the format we expect
        # TODO: don't hardcode the dictionary
        tag_map = {u'OK': 1, u'BAD': 0}
        train_tags = [[tag_map[tag] for tag in seq] for seq in train_tags]
        test_tags = [[tag_map[tag] for tag in seq] for seq in test_tags]

        # make sure that everything is numpy
        # cast the dataset to numpy array (ndarrays)
        # note that these are _NOT_ matrices, because the inner sequences have different lengths
        x_train = np.array([np.array(xi) for xi in train_features])
        y_train = np.array([np.array(xi) for xi in train_tags])
        x_test = np.array([numpy.array(xi) for xi in test_features])
        y_test = np.array([numpy.array(xi) for xi in test_tags])

        # SEQLEARN
        # from seqlearn.perceptron import StructuredPerceptron
        #
        # # seqlearn needs a flat list of instances
        # x_train = np.array([i for seq in x_train for i in seq])
        # y_train = np.array([i for seq in y_train for i in seq])
        # x_test = np.array([i for seq in x_test for i in seq])
        # y_test = np.array([i for seq in y_test for i in seq])
        #
        # # seqlearn requires the lengths of each sequence
        # lengths_train = [len(seq) for seq in train_features]
        # lengths_test = [len(seq) for seq in test_features]
        #
        # clf = StructuredPerceptron(verbose=True, max_iter=400)
        # clf.fit(x_train, y_train, lengths_train)
        #
        # structured_predictions = clf.predict(x_test, lengths_test)
        # logger.info('f1 from seqlearn: {}'.format(f1_score(y_test, structured_predictions, average=None)))
        # ipdb.set_trace()

        # END SEQLEARN

        # pystruct
        from marmot.learning.pystruct_sequence_learner import PystructSequenceLearner
        sequence_learner = PystructSequenceLearner()
        sequence_learner.fit(x_train, y_train)
        structured_hyp = sequence_learner.predict(x_test)

        logger.info('scoring sequential model...')
        # print('score: ' + str(structured_predictor.score(x_test, y_test)))

        # TODO: implement this in the config
        # classifier_type = import_class(config['learning']['classifier']['module'])

        # TODO: the flattening is currently a hack to let us use the same evaluation code for structured and plain tasks
        flattened_hyp = flatten(structured_hyp)

        # end pystruct

        test_predictions = flattened_hyp
        flattened_ref = flatten(y_test)
        test_tags = flattened_ref

        logger.info('Structured prediction f1: ')
        print(f1_score(flattened_ref, flattened_hyp, average=None))

    else:
        # data_type is 'token' or 'plain'
        logger.info('start training...')
        classifier_type = import_class(config['learning']['classifier']['module'])
        # train the classifier(s)
        classifier_map = map_classifiers(train_features, train_tags, classifier_type, data_type=data_type)
        logger.info('classifying the test instances')
        test_predictions = predict_all(test_features, classifier_map, data_type=data_type)

    # TODO: this section only works for 'plain'

    print(f1_score(test_predictions, test_tags, average=None))

    # EVALUATION
    logger.info('evaluating your results')

    # TODO: remove the hard coding of the tags here
    bad_count = sum(1 for t in test_tags if t == u'BAD' or t == 0)
    good_count = sum(1 for t in test_tags if t == u'OK' or t == 1)

    total = len(test_tags)
    ipdb.set_trace()
    assert (total == bad_count+good_count), 'tag counts should be correct'
    percent_good = good_count / total
    logger.info('percent good in test set: {}'.format(percent_good))
    logger.info('percent bad in test set: {}'.format(1 - percent_good))

    import numpy as np

    random_class_results = []
    random_weighted_results = []
    for i in range(20):
        random_tags = list(np.random.choice([u'OK', u'BAD'], total, [percent_good, 1-percent_good]))
        # random_tags = [u'GOOD' for i in range(total)]
        random_class_f1 = f1_score(test_tags, random_tags, average=None)
        random_class_results.append(random_class_f1)
        # logger.info('two class f1 random score ({}): {}'.format(i, random_class_f1))
        # random_average_f1 = f1_score(random_tags, test_tags, average='weighted')
        random_average_f1 = weighted_fmeasure(test_tags, random_tags)
        random_weighted_results.append(random_average_f1)
        # logger.info('average f1 random score ({}): {}'.format(i, random_average_f1))

    avg_random_class = np.average(random_class_results, axis=0)
    avg_weighted = np.average(random_weighted_results)
    logger.info('two class f1 random average score: {}'.format(avg_random_class))
    logger.info('weighted f1 random average score: {}'.format(avg_weighted))

    actual_class_f1 = f1_score(test_tags, test_predictions, average=None)
    actual_average_f1 = weighted_fmeasure(test_tags, test_predictions)
    logger.info('two class f1 ACTUAL SCORE: {}'.format(actual_class_f1))
    logger.info('weighted f1 ACTUAL SCORE: {}'.format(actual_average_f1))

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
        # logger.info('F1 score: ')
        # print(f1)

    # write_res_to_file(config['test']['output'], test_predictions)

    # END EVALUATION


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("configuration_file", action="store", help="path to the config file (in YAML format).")
    args = parser.parse_args()
    experiment_config = {}

    # Experiment hyperparams
    cfg_path = args.configuration_file
    # read configuration file
    with open(cfg_path, "r") as cfg_file:
        config = yaml.load(cfg_file.read())
    main(experiment_config)

