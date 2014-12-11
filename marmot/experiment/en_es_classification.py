from argparse import ArgumentParser
import yaml
import os, sys
import logging
import marmot

from marmot.experiment import learning_utils
from marmot.experiment.experiment_utils import *

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

    interesting_tokens = import_and_call_function(config['interesting_tokens'])
    print "INTERESTING TOKENS: ", interesting_tokens
#    sys.exit(2)
    logger.info('The number of interesting tokens is: ' + str(len(interesting_tokens)))
    workers = config['workers']

    # Note: context creators currently create their own interesting tokens internally (interesting tokens controls the index of the context creator)
    logger.info('building the context creators...')
    train_context_creators = build_context_creators(config['context_creators'])

    # get the contexts for all of our interesting words (may be +,- or, multi-class)
    logger.info('mapping the training contexts over the interesting tokens in train...')
    train_contexts = map_contexts(interesting_tokens, train_context_creators, workers=workers)
    # --> map context extraction over a multiprocessing Pool (divide our interesting words between threads)
    # --> check speed difference between multiprocessing Pool and multiprocessing.dummy ThreadPool
    # from multiprocessing import Pool
    # from multiprocessing.dummy import Pool as ThreadPool

    # load and parse the test data
    logger.info('mapping the training contexts over the interesting tokens in test...')
    test_context_creator = build_context_creator(config['testing'])
    test_contexts = map_contexts(interesting_tokens, [test_context_creator])

    min_total = config['filters']['min_total']
    # filter token contexts based on the user-specified filer criteria
    logger.info('filtering the contexts by the total number of available instances...')
    train_contexts = filter_contexts(train_contexts, min_total=min_total)
    test_contexts = filter_contexts(test_contexts, min_total=min_total)

    # make sure the test_context and train_context keys are in sync
    sync_keys(train_contexts, test_contexts)

    # test_contexts = filter_contexts(test_contexts, min_total=min_total)
    assert set(test_contexts.keys()) == set(train_contexts.keys())

#    print train_contexts.items()[0]
#    sys.exit(2) 

    # extract the 'tag' attribute into the y-value for classification
    wmt_binary_classes = {u'BAD': 0, u'OK': 1}
    train_context_tags = tags_from_contexts(train_contexts)
    train_context_tags = { k:np.array( [wmt_binary_classes[v] for v in val] ) for k, val in train_context_tags.items() }
    print 'TRAIN TAGS: ', train_context_tags

    # tags may need to be converted to be consistent with the training data
    test_contexts = convert_tagset(wmt_binary_classes, test_contexts)
    test_tags_actual = tags_from_contexts(test_contexts)
    print 'TEST TAGS: ', test_tags_actual

    # all of the feature extraction should be parallelizable
    # note that a feature extractor MUST be able to parse the context exchange format, or it should throw an error:
    # { 'token': <token>, index: <idx>, 'source': [<source toks>]', 'target': [<target toks>], 'tag': <tag>}
    feature_extractors = build_feature_extractors(config['feature_extractors'])
    logger.info('mapping the feature extractors over the contexts for test...')
    test_context_features = contexts_to_features_categorical(test_contexts, feature_extractors, workers=1)
    logger.info('mapping the feature extractors over the contexts for train...')
    train_context_features = contexts_to_features_categorical(train_contexts, feature_extractors, workers=workers)

#    print "TEST and TRAIN CONTEXT LENGTH: ", len(test_context_features), len(train_context_features)
#    print "TEST flattened: ", flatten(test_context_features.values())
#    print "TRAIN flattened: ", flatten(train_context_features.values())
#    print "BOTH: ", 
    all_values = flatten(test_context_features.values())
    all_values.extend( flatten(train_context_features.values()) )
    binarizers = fit_binarizers( all_values )
    test_context_features = { k:[binarize(v, binarizers) for v in val] for k, val in test_context_features.items() }
    train_context_features = { k:[binarize(v, binarizers) for v in val] for k, val in train_context_features.items() }

#    print test_context_features.items()[0]
#    print train_context_features.items()[0]
#    sys.exit(2)

    # BEGIN LEARNING
    classifier_type = import_class(config['learning']['classifier']['module'])
    # train the classifier for each token
    classifier_map = learning_utils.token_classifiers(train_context_features, train_context_tags, classifier_type)

    # classify the test instances
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



    # Working: use the WMT evaluation script (modify as necessary)
    # get data points for visualization!


    # cross validation experiments to test if our chosen classifier is any good at predicting our data
    # note -- this method only makes sense when there are enough test instances for a given token (i.e. we don't want to do cross validation with only 3 instances)
    # from sklearn.cross_validation import StratifiedKFold, permutation_test_score
    # import matplotlib.pyplot as plt
    # from marmot.experiment.learning_utils import init_classifier
    # logger.info('TRYING CROSS-VAL P VALUE - evaluated on the training contexts')
    # for token, features in train_context_features.iteritems():
    #     classifier_obj = init_classifier(classifier_type)
    #
    #     y = train_context_tags[token]
    #     X = features
    #     cv = StratifiedKFold(y, 2)
    #     score, permutation_scores, pvalue = permutation_test_score(classifier_obj, X, y, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=4)
    #
    #     logger.info('TOKEN IS: ' + token)
    #     logger.info('Num instances: ' + str(len(X)))
    #     logger.info('num negative: ' + str(np.sum(y == 0)))
    #     logger.info('num positive: ' + str(np.sum(y == 1)))
    #     logger.info("\tClassification score %s (pvalue : %s)" % (score, pvalue))
    #
    #     # View histogram of permutation scores
    #     n_classes = np.unique(y).size
    #     plt.hist(permutation_scores, 20, label='Permutation scores')
    #     ylim = plt.ylim()
    #     plt.plot(2 * [score], ylim, '--g', linewidth=3,
    #              label='Classification Score'
    #              ' (pvalue %s)' % pvalue)
    #
    #     # TODO: this is luck WITHOUT using the prior
    #     plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')
    #
    #     plt.ylim(ylim)
    #     plt.legend()
    #     plt.xlabel('Score')
    #     # plt.savefig('test_data/plots/' + token, orientation='landscape')
    #     # plsavefig(fname, dpi=None, facecolor='w', edgecolor='w',
    #     # orientation='portrait', papertype=None, format=None,
    #     # transparent=False, bbox_inches=None, pad_inches=0.1,
    #     # frameon=None)
    #
    #     plt.show()


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

