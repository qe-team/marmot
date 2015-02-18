from __future__ import print_function, division

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
logger = logging.getLogger('experiment_logger')


def main(config):
    workers = config['workers']

    # unify data representations
    # test_data and training_data - lists of dictionaries { target: target_file, source: source_file, tags: tags).
    # <tags> can be a file with tags or a single tag

    # build objects for additional representations
    train_representation_generators = build_objects(config['representations']['training'])
    test_representation_generators = build_objects(config['representations']['test'])

    # get additional representations
    # generators return a pair (label, representation)
    # TODO: generators should return data, not filenames
    # TODO: generators can check if the file they are trying to make already exists, if so, they should read it, if not, build the representation, and persist according to a flag
    train_data = {}
    test_data = {}
    for r in train_representation_generators:
        train_data = r.generate(train_data)
    for r in test_representation_generators:
        test_data = r.generate(test_data)

    logger.info('here are your representations: {}'.format(train_data.keys()))

    # TODO: since there is only one context creator and it does nothing, we don't need it any more

    # how to generate the old {token:context_list} representation?
    # Answer: we should do this in 'create_contexts'
    data_type = config['contexts'] if 'contexts' in config else 'plain'

    # TODO: `create_contexts` means 'read the files in the representation generator object'

    # TODO: files are implicitly whitespace tokenized
    # TODO: create_contexts maps a whitespace tokenized, line by line dataset into one of our three data representations
    test_contexts = create_contexts(test_data, data_type=data_type)
    train_contexts = create_contexts(train_data, data_type=data_type)

#    for r in representation_generators:
#        r.cleanup()

    # make sure the test_context and train_context keys are in sync
    # TODO: this is important when we are learning token-level classifiers
#    experiment_utils.sync_keys(train_contexts, test_contexts)

    # TODO: this is important when we are learning token-level classifiers
    # test_contexts = filter_contexts(test_contexts, min_total=min_total)
#    assert set(test_contexts.keys()) == set(train_contexts.keys())

    # TODO: error here for sequential data
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

    logger.info('number of training instances: {}'.format(len(train_features)))
    logger.info('number of testing instances: {}'.format(len(test_features)))

    # flatten so that we can properly binarize the features
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

    logger.info('training sets successfully generated')

    # learning
    from sklearn.metrics import f1_score
    if data_type == 'sequential':
        # raise NotImplementedError('sequential learning hasnt been implemented yet')

        logger.info('training sequential model...')
        from pystruct.models import ChainCRF
        # from pystruct.models import EdgeFeatureGraphCRF

        from pystruct.learners import OneSlackSSVM
        from pystruct.learners import StructuredPerceptron
        from pystruct.learners import SubgradientSSVM

        # Train linear chain CRF
        model = ChainCRF(directed=True)
        # structured_predictor = OneSlackSSVM(model=model, C=.1, inference_cache=50, tol=0.1, n_jobs=1)
        structured_predictor = StructuredPerceptron(model=model, average=True)

        # map tags to ints
        tag_map = {u'OK': 1, u'BAD': 0}
        train_tags = [[tag_map[tag] for tag in seq] for seq in train_tags]
        test_tags = [[tag_map[tag] for tag in seq] for seq in test_tags]

        x_train = numpy.array([numpy.array(xi) for xi in train_features])
        y_train = numpy.array([numpy.array(xi) for xi in train_tags])
        x_test = numpy.array([numpy.array(xi) for xi in test_features])
        y_test = numpy.array([numpy.array(xi) for xi in test_tags])
        structured_predictor.fit(x_train, y_train)
        logger.info('scoring sequential model...')
        print('score: ' + str(structured_predictor.score(x_test, y_test)))
        # classifier_type = import_class(config['learning']['classifier']['module'])
        structured_hyp = structured_predictor.predict(x_test)
        flattened_hyp = flatten(structured_hyp)
        flattened_ref = flatten(y_test)
        logger.info('f1: ')
        print(f1_score(flattened_ref, flattened_hyp, average=None))


    else:
        # data_type is 'token' or 'plain'
        logger.info('start training...')
        classifier_type = import_class(config['learning']['classifier']['module'])
        # train the classifier(s)
        classifier_map = map_classifiers(train_features, train_tags, classifier_type, data_type=data_type)

    # Chris: commented for sequential learning
    # TODO: this section only works for 'plain'
    logger.info('classifying the test instances')
    test_predictions = predict_all(test_features, classifier_map, data_type=data_type)
    print(f1_score(test_predictions, test_tags, average=None))
    # Chris: commented for sequential learning

    logger.info('evaluating your results')
    bad_count = sum(1 for t in test_tags if t == u'BAD')
    good_count = sum(1 for t in test_tags if t == u'OK')

    total = len(test_tags)
    assert (total == bad_count+good_count), 'tag counts should be correct'
    percent_good = good_count / total
    logger.info('percent good in test set: {}'.format(percent_good))
    logger.info('percent bad in test set: {}'.format(1 - percent_good))

    import numpy as np

    random_class_results = []
    random_weighted_results = []
    for i in range(20):
        random_tags = list(np.random.choice([u'OK', u'BAD'], total, [percent_good, 1-percent_good]))
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

