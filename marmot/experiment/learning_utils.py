# utils for interfacing with Scikit-Learn
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('testlogger')

# TODO: allow specification of cross-validation params at init time
def init_classifier(classifier_type, args=None):
    if args is not None:
        return classifier_type(*args)
    return classifier_type()

def train_classifier(X, y, classifier):
    classifier.fit(X,y)

def token_classifiers(token_contexts, token_tags, classifier_type, classifier_args=None):
    classifier_map = {}
    for token, contexts in token_contexts.items():
        logger.info('training classifier for token: {}'.format(token.encode('utf-8')))
        token_classifier = init_classifier(classifier_type, classifier_args)
        token_classifier.fit(contexts, token_tags[token])
        classifier_map[token] = token_classifier

    return classifier_map

