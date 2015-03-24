from __future__ import print_function, division

from argparse import ArgumentParser
import yaml
import logging

from marmot.experiment.import_utils import *
from marmot.experiment.preprocessing_utils import *
from marmot.evaluation.evaluation_utils import compare_vocabulary
from marmot.experiment.learning_utils import feature_selection

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('experiment_logger')


def main(config):
    workers = config['workers']
    train_data_generators = build_objects(config['datasets']['training'])
    train_data = {}
    for gen in train_data_generators:
        data = gen.generate()
        for key in data:
            if key not in train_data:
                train_data[key] = []
            train_data[key].extend(data[key])
    # test
    test_data_generator = build_object(config['datasets']['test'][0])
    test_data = test_data_generator.generate()

    # additional representations
    if 'representations' in config:
        representation_generators = build_objects(config['representations'])
    else:
        representation_generators = []
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
    feature_names = [n for extractor in feature_extractors for n in extractor.get_feature_names()] 

    new_features, old_max, new_max = feature_selection(train_features, train_tags, test_features, test_tags, feature_names, data_type='sequential')
    logger.info("Feature selection done. Best-performing features subset:\n\n {}".format('\n'.join(new_features)))
    logger.info("Result with all features: {}, with chosen features: {}".format(old_max, new_max))


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("configuration_file", action="store", help="path to the config file (in YAML format).")
    args = parser.parse_args()
    experiment_config = {}

    # Experiment hyperparams
    cfg_path = args.configuration_file
    # read configuration file
    with open(cfg_path, "r") as cfg_file:
        experiment_config = yaml.load(cfg_file.read())
        main(experiment_config)
