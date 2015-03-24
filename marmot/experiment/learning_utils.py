# utils for interfacing with Scikit-Learn
import logging
import numpy as np
import copy
from multiprocessing import Pool

from sklearn.metrics import f1_score
from marmot.learning.pystruct_sequence_learner import PystructSequenceLearner
from marmot.experiment.import_utils import call_for_each_element
from marmot.experiment.preprocessing_utils import flatten, fit_binarizers, binarize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('testlogger')


# TODO: allow specification of cross-validation params at init time
def init_classifier(classifier_type, args=None):
    if args is not None:
        return classifier_type(*args)
    return classifier_type()


def train_classifier(X, y, classifier):
    classifier.fit(X, y)


def map_classifiers(all_contexts, tags, classifier_type, data_type='plain', classifier_args=None):
    if data_type == 'plain':
        assert(type(all_contexts) == np.ndarray or type(all_contexts) == list)
        logger.info('training classifier')
        classifier = init_classifier(classifier_type, classifier_args)
        classifier.fit(all_contexts, tags)
        return classifier
    elif data_type == 'token':
        assert(type(all_contexts) == dict)
        classifier_map = {}
        for token, contexts in all_contexts.items():
            logger.info('training classifier for token: {}'.format(token.encode('utf-8')))
            token_classifier = init_classifier(classifier_type, classifier_args)
            token_classifier.fit(contexts, tags[token])
            classifier_map[token] = token_classifier
        return classifier_map


def predict_all(test_features, classifier_map, data_type='plain'):
    if data_type == 'plain':
        predictions = classifier_map.predict(test_features)
        return predictions
    elif data_type == 'token':
        test_predictions = {}
        for key, features in test_features.iteritems():
            try:
                classifier = classifier_map[key]
                predictions = classifier.predict(features)
                test_predictions[key] = predictions
            except KeyError as e:
                print(key + " - is NOT in the classifier map")
                raise
        return test_predictions


def run_prediction((train_data, train_tags, test_data, test_tags, idx)):
    logger.info('training sequential model...')
    all_values = flatten(train_data)
    # binarize
    binarizers = fit_binarizers(all_values)
    test_data = call_for_each_element(test_data, binarize, [binarizers], data_type='sequential')
    train_data = call_for_each_element(train_data, binarize, [binarizers], data_type='sequential')

    x_train = np.array([np.array(xi) for xi in train_data])
    y_train = np.array([np.array(xi) for xi in train_tags])
    x_test = np.array([np.array(xi) for xi in test_data])
    y_test = np.array([np.array(xi) for xi in test_tags])
    
    sequence_learner = PystructSequenceLearner()
    sequence_learner.fit(x_train, y_train)
    structured_hyp = sequence_learner.predict(x_test)
    
    logger.info('scoring sequential model...')
    flattened_hyp = flatten(structured_hyp)
    
    flattened_ref = flatten(y_test)
    test_tags = flattened_ref
    
    logger.info('Structured prediction f1: ')
    cur_res = f1_score(flattened_ref, flattened_hyp, average=None)
    logger.info('[ {}, {} ], {}'.format(cur_res[0], cur_res[1], f1_score(flattened_ref, flattened_hyp, pos_label=None)))

    return (cur_res, idx)


# remove the feature number <idx>
def get_reduced_set(features_list, idx):
    new_features_list = [obj[:idx] + obj[idx+1:] for obj in features_list]
    return new_features_list


# train the model on all combinations of the feature set without one element
# TODO: the target metric should be tunable (now the f1 score of BAD class)
def selection_epoch(old_result, train_data, train_tags, test_data, test_tags, feature_names, data_type='sequential'):
    reduced_res = np.zeros((len(feature_names),))
    max_res = old_result
    reduced_train = train_data
    reduced_test = test_data
    reduced_features = feature_names
    for idx, name in enumerate(feature_names):
        logger.info("Excluding feature {}".format(name))
        # new feature sets without the feature <idx>
        cur_reduced_train = call_for_each_element(train_data, get_reduced_set, args=[idx], data_type=data_type)
        cur_reduced_test = call_for_each_element(test_data, get_reduced_set, args=[idx], data_type=data_type)

        # train a sequence labeller
        if data_type == 'sequential':
            cur_res = run_prediction((cur_reduced_train, train_tags, cur_reduced_test, test_tags, idx))
            reduced_res[idx] = cur_res[0]
            # if the result is better than previous -- save as maximum
            if cur_res[0] > max_res:
                max_res = cur_res[0]
                reduced_train = cur_reduced_train
                reduced_test = cur_reduced_test
                reduced_features = feature_names[:idx] + feature_names[idx+1:]

    # if better result is found -- return it
    if max_res > old_result:
        return (idx, max_res, reduced_train, reduced_test, reduced_features)
    # none of the reduced sets worked better
    else:
        return (-1, old_result, [], [], [])


def selection_epoch_multi(old_result, train_data, train_tags, test_data, test_tags, feature_names, workers, data_type='sequential'):
#    reduced_res = np.zeros((len(feature_names),))
    max_res = old_result
    reduced_train = train_data
    reduced_test = test_data
    reduced_features = feature_names
    parallel_data = []
    for idx, name in enumerate(feature_names):
        # new feature sets without the feature <idx>
        cur_reduced_train = call_for_each_element(train_data, get_reduced_set, args=[idx], data_type=data_type)
        cur_reduced_test = call_for_each_element(test_data, get_reduced_set, args=[idx], data_type=data_type)
        parallel_data.append((cur_reduced_train, train_tags, cur_reduced_test, test_tags, idx))

    # train a sequence labeller
    if data_type == 'sequential':
        pool = Pool(workers)
        reduced_res = pool.map(run_prediction, parallel_data)
        print "Multiprocessing output: ", reduced_res

    all_res = [res[0][0] for res in reduced_res]
    # some feature set produced better result
    if max(all_res) > old_result:
        odd_feature_num = reduced_res[np.argmax(all_res)][1]
        reduced_train = call_for_each_element(train_data, get_reduced_set, args=[odd_feature_num], data_type=data_type)
        reduced_test = call_for_each_element(test_data, get_reduced_set, args=[odd_feature_num], data_type=data_type)
        reduced_features = feature_names[:odd_feature_num] + feature_names[odd_feature_num+1:]
        logger.info("Old result: {}, new result: {}, removed feature is {}".format(old_result, max(all_res), feature_names[odd_feature_num]))
        return (feature_names[odd_feature_num], max(all_res), reduced_train, reduced_test, reduced_features)
    # none of the reduced sets worked better
    else:
        logger.info("No improvement on this round")
        return ("", old_result, [], [], [])


def feature_selection(train_data, train_tags, test_data, test_tags, feature_names, data_type='sequential'):

    tag_map = {u'OK': 1, u'BAD': 0}
    train_tags = [[tag_map[tag] for tag in seq] for seq in train_tags]
    test_tags = [[tag_map[tag] for tag in seq] for seq in test_tags]

    full_set_result = run_prediction((train_data, train_tags, test_data, test_tags, 0))

    logger.info("Feature selection")
    odd_feature = None
    baseline_res = full_set_result[0][0]
    logger.info("Baseline result: {}".format(baseline_res))
    reduced_train = copy.deepcopy(train_data)
    reduced_test = copy.deepcopy(test_data)
    reduced_features = copy.deepcopy(feature_names)
    odd_feature_list = []
    # reduce the feature set while there are any combinations that give better result
    cnt = 1
    old_max = baseline_res
    while odd_feature != "" and len(reduced_features) > 1:
        logger.info("Feature selection: round {}".format(cnt))
        odd_feature, max_res, reduced_train, reduced_test, reduced_features = selection_epoch_multi(old_max, reduced_train, train_tags, reduced_test, test_tags, reduced_features, 10, data_type=data_type)
#        odd_feature, reduced_train, reduced_test, reduced_features = selection_epoch(old_max, reduced_train, train_tags, reduced_test, test_tags, reduced_features, data_type=data_type)
        odd_feature_list.append(odd_feature)
        old_max = max_res
        cnt += 1

    # form a set of reduced feature names and feature numbers
    new_feature_list = []
    for feature in feature_names:
        if feature not in odd_feature_list:
            new_feature_list.append(feature)
    logger.info("Feature selection is terminating, good features are: {}".format(' '.join(new_feature_list)))

    return (new_feature_list, baseline_res, old_max)
