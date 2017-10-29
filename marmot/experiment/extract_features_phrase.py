from __future__ import print_function, division

from argparse import ArgumentParser
import yaml
import logging
import os

from marmot.experiment.import_utils import build_objects, build_object, mk_tmp_dir, call_for_each_element
from marmot.experiment.preprocessing_utils import tags_from_contexts, contexts_to_features
from marmot.evaluation.evaluation_utils import compare_vocabulary
from marmot.util.persist_features import persist_features
from marmot.util.generate_crf_template import generate_crf_template
from marmot.experiment.context_utils import create_contexts_ngram, get_contexts_words_number

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('experiment_logger')
'''
Only feature extraction
Extract features and save in CRF++ format
'''


def main(config):
    workers = config['workers']
    tmp_dir = config['tmp_dir']
    tmp_dir = mk_tmp_dir(tmp_dir)

    # REPRESENTATION GENERATION
    # main representations (source, target, tags)
    dev, test = False, False
    # training
    if 'training' in config['datasets']:
        train_data_generator = build_object(config['datasets']['training'][0])
        train_data = train_data_generator.generate()
    # test
    if 'test' in config['datasets']:
        test = True
        test_data_generator = build_object(config['datasets']['test'][0])
        test_data = test_data_generator.generate()
    # dev
    if 'dev' in config['datasets']:
        dev = True
        dev_data_generator = build_object(config['datasets']['dev'][0])
        dev_data = dev_data_generator.generate()
    # additional representations
    if 'representations' in config:
        representation_generators = build_objects(config['representations'])
    else:
        representation_generators = []
    for r in representation_generators:
        train_data = r.generate(train_data)
        if test:
            test_data = r.generate(test_data)
        if dev:
            dev_data = r.generate(dev_data)

    logger.info("Simple representations: {}".format(len(train_data['target'])))
    logger.info('here are the keys in your representations: {}'.format(train_data.keys()))

    # the data_type is the format corresponding to the model of the data that the user wishes to learn
    data_type = 'sequential'

    bad_tagging = config['bad_tagging']
    tags_format = config['tags_format'] if 'tags_format' in config else 'word'
    train_contexts = create_contexts_ngram(train_data, data_type=data_type, test=False, bad_tagging=bad_tagging, unambiguous=config['unambiguous'], tags_format=tags_format)
    if test:
        test_contexts = create_contexts_ngram(test_data, data_type=data_type, test=True, bad_tagging=bad_tagging, unambiguous=config['unambiguous'], tags_format=tags_format)
    if dev:
        dev_contexts = create_contexts_ngram(dev_data, data_type=data_type, test=True, bad_tagging=bad_tagging, unambiguous=config['unambiguous'], tags_format=tags_format)

    logger.info('Vocabulary comparison -- coverage for each dataset: ')
    logger.info(compare_vocabulary([train_data['target'], test_data['target']]))

    # END REPRESENTATION GENERATION

    # FEATURE EXTRACTION
    train_tags = call_for_each_element(train_contexts, tags_from_contexts, data_type=data_type)
    if test:
        test_tags = call_for_each_element(test_contexts, tags_from_contexts, data_type=data_type)
    if dev:
        dev_tags = call_for_each_element(dev_contexts, tags_from_contexts, data_type=data_type)

    # word-level tags and phrase lengths
    if test:
        test_phrase_lengths = [get_contexts_words_number(cont) for cont in test_contexts]
    if dev:
        dev_phrase_lengths = [get_contexts_words_number(cont) for cont in dev_contexts]

    logger.info('creating feature extractors...')
    feature_extractors = build_objects(config['feature_extractors'])
    if test:
        logger.info('mapping the feature extractors over the contexts for test...')
        test_features = call_for_each_element(test_contexts, contexts_to_features, [feature_extractors, workers], data_type=data_type)
    if dev:
        logger.info('mapping the feature extractors over the contexts for dev...')
        dev_features = call_for_each_element(dev_contexts, contexts_to_features, [feature_extractors, workers], data_type=data_type)
    logger.info('mapping the feature extractors over the contexts for train...')
    train_features = call_for_each_element(train_contexts, contexts_to_features, [feature_extractors, workers], data_type=data_type)

    logger.info('number of training instances: {}'.format(len(train_features)))
    logger.info('number of testing instances: {}'.format(len(test_features)))

    logger.info('All of your features now exist in their raw representation, but they may not be numbers yet')
    # END FEATURE EXTRACTION

    # persisting features
    logger.info('training and test sets successfully generated')

    experiment_datasets = [{'name': 'train', 'features': train_features, 'tags': train_tags, 'phrase_lengths': None}]
    if test:
        experiment_datasets.append({'name': 'test', 'features': test_features, 'tags': test_tags, 'phrase_lengths': test_phrase_lengths})
    if dev:
        experiment_datasets.append({'name': 'dev', 'features': dev_features, 'tags': dev_tags, 'phrase_lengths': dev_phrase_lengths})
    feature_names = [f for extractor in feature_extractors for f in extractor.get_feature_names()]

    persist_dir = config['persist_dir'] if 'persist_dir' in config else tmp_dir
    persist_dir = mk_tmp_dir(persist_dir)
    persist_format = config['persist_format']
    logger.info('persisting your features to: {}'.format(persist_dir))
    # for each dataset, write a file and persist the features
    for dataset_obj in experiment_datasets:
        persist_features(dataset_obj['name'],
                         dataset_obj['features'],
                         persist_dir,
                         feature_names=feature_names,
                         phrase_lengths=dataset_obj['phrase_lengths'],
                         tags=dataset_obj['tags'],
                         file_format=persist_format)
    # generate a template for CRF++ feature extractor
    feature_num = len(feature_names)
    if persist_format == 'crf++':
        generate_crf_template(feature_num, 'template', persist_dir)

    logger.info('Features persisted to: {}'.format(', '.join([os.path.join(persist_dir, nn) for nn in [obj['name'] for obj in experiment_datasets]])))


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("configuration_file", action="store", help="path to the config file (in YAML format).")
    parser.add_argument("--data_type", help="data type - sequential or plain")
    parser.add_argument("--bad_tagging", help="tagging -- optimistic, pessimistic or super-pessimistic")
    parser.add_argument("--unambiguous", default=0, help="make the tagging unambiguous -- no segmentation for spans of BAD tag (values - 0 or 1, default 0)")
    parser.add_argument("--output_name", default="output", help="file to store the test set tagging")

    args = parser.parse_args()
    experiment_config = {}

    # Experiment hyperparams
    cfg_path = args.configuration_file
    # read configuration file
    with open(cfg_path, "r") as cfg_file:
        experiment_config = yaml.load(cfg_file.read())

    if args.data_type is not None:
        experiment_config['data_type'] = args.data_type
    if args.bad_tagging is not None:
        experiment_config['bad_tagging'] = args.bad_tagging
    experiment_config['unambiguous'] = True if int(args.unambiguous) == 1 else False
    experiment_config['output_name'] = args.output_name

    main(experiment_config)
