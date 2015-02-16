# utils for interfacing with Scikit-Learn
import logging
import numpy as np
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

