from __future__ import print_function, division

from argparse import ArgumentParser
import os
import sys
import yaml
import time
import logging
from subprocess import call
from sklearn.metrics import f1_score
from marmot.experiment.import_utils import call_for_each_element, build_object, build_objects, mk_tmp_dir
from marmot.experiment.preprocessing_utils import create_contexts, tags_from_contexts, contexts_to_features
from marmot.util.persist_features import persist_features
from marmot.util.add_bigram_features import add_bigram_features

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('experiment_logger')


def load_features(features_file, feature_names_file=None):
    feature_names = []
    if feature_names_file is not None:
        for n_line in open(feature_names_file):
            feature_names.append(n_line.decode('utf-8').strip('\n'))
    features = []
    cur_features = []
    for f_line in open(features_file):
        if f_line == '\n':
            features.append(cur_features)
            cur_features = []
            continue
        f_line = f_line.decode('utf-8').strip('\n')
        f_chunks = f_line.split("\t")
        for f in f_chunks:
            try:
                cur_features.append(float(f))
            except ValueError:
                cur_features.append(f)
    if len(cur_features) != 0:
        features.append(cur_features)
    return features, feature_name


# load labels (one line per sentence, OK/BAD)
def load_tags(tags_file, data_type):
    tags = []
    for line in open(tags_file):
        if data_type == "plain":
            tags.extend(line.strip("\n").split())
        elif data_type == "sequential":
            tags.append(line.strip("\n").split())
        else:
            print("Unknown data type: {}".format(data_type))
            sys.exit()
    return tags    


# parse SVLight output,
# return the predicted tags (0 - BAD, 1 - GOOD)
def get_test_score_blind(test_file):
    predicted = []
    tag_map = {'+1': 1, '-1': 0}
    for line in open(test_file):
        label = line[line.find(':')+1:line.find(' ')]
        predicted.append(tag_map[label])
    return predicted


def main(config):
    workers = config['workers']
    tmp_dir = config['tmp_dir'] if 'tmp_dir' in config else None
    tmp_dir = mk_tmp_dir(tmp_dir)
    time_stamp = str(time.time())

    #----------------------Feature extraction from file------------------
    if 'pre-extracted' in config:
        train_features, feature_names = load_features(config['pre-extracted']['train-features'], config['pre-extracted']['feature-names'])
        test_features, _ = load_features(config['pre-extracted']['test-features'])
        train_tags = load_tags(config['pre-extracted']['train-tags']
        test_tags = load_tags(config['pre-extracted']['test-tags']

    #--------------REPRESENTATION GENERATION---------------------
    else:
        # main representations (source, target, tags)
        # training
        train_data_generator = build_object(config['datasets']['train'][0])
        train_data = train_data_generator.generate()
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

        logger.info("Train data keys: {}".format(train_data.keys()))
        logger.info("Train data sequences: {}".format(len(train_data['target'])))
        logger.info("Sample sequence: {}".format([w.encode('utf-8') for w in train_data['target'][0]]))

        # the data_type is the format corresponding to the model of the data that the user wishes to learn
        data_type = config['data_type'] if 'data_type' in config else 'plain'

        test_contexts = create_contexts(test_data, data_type=data_type)
        train_contexts = create_contexts(train_data, data_type=data_type)

        #------------------------FEATURE EXTRACTION--------------------------
        train_tags = call_for_each_element(train_contexts, tags_from_contexts, data_type=data_type)
        test_tags = call_for_each_element(test_contexts, tags_from_contexts, data_type=data_type)

        # create features
        logger.info('creating feature extractors...')
        feature_extractors = build_objects(config['feature_extractors'])
        logger.info('mapping the feature extractors over the contexts for test...')
        test_features = call_for_each_element(test_contexts, contexts_to_features, [feature_extractors, workers], data_type=data_type)
        logger.info('mapping the feature extractors over the contexts for train...')
        train_features = call_for_each_element(train_contexts, contexts_to_features, [feature_extractors, workers], data_type=data_type)

        feature_names = [f for extractor in feature_extractors for f in extractor.get_feature_names()]

    if 'bigram_features' in config and config['bigram_features']:
        train_features = call_for_each_element(train_features, add_bigram_features, [train_tags], data_type=data_type)
        train_features = call_for_each_element(train_features, add_bigram_features, [train_tags], data_type=data_type)

    # create binary features for training
    logger.info('number of training instances: {}'.format(len(train_features)))
    logger.info('number of testing instances: {}'.format(len(test_features)))

    #-------------------PERSIST FEATURES--------------------------------

    if config['features']['persist_dir']:
        persist_dir = config['features']['persist_dir']
    else:
        persist_dir = os.path.getcwd()
    logger.info('persisting your features to: {}'.format(persist_dir))
    if data_type == 'plain':
        persist_format = 'svm_light'
    elif data_type == 'sequential':
        persist_format = 'crf_suite'
    # for each dataset, write a file and persist the features
    train_file_name = persist_features("train", train_features, persist_dir, feature_names=feature_names, tags=train_tags, file_format=persist_format)
    test_file_name = persist_features("test", test_features, persist_dir, feature_names=feature_names, tags=test_tags, file_format=persist_format)
    test_output = os.path.join(persist_dir, 'out')

    #---------------------------TRAINING---------------------------------
    #----------------------------SVM LIGHT-------------------------------
    if data_type == 'plain':
        kernel = 0
        if 'svm_params' in config:
            try:
                kernel = int(config['svm_params']['kernel'])
            except ValueError:
                kernel = 0
            kernel = kernel if kernel <= 4 else 0
        model_name = os.path.join(tmp_dir, 'svmlight_model_file'+time_stamp)
        call(['/export/tools/varvara/svm_multiclass/svm_light/svm_learn', '-t', str(kernel), train_file_name, model_name])
        logger.info("Training completed, start testing")
        call(['/export/tools/varvara/svm_multiclass/svm_light/svm_classify', '-f', '0', test_file_name, model_name, test_output])
        logger.info("Testing completed")
        predicted = get_test_score_blind(test_output)
        tag_map = {'OK': 1, 'BAD': 0}
        test_tags_num = [tag_map[t] for t in test_tags]
        logger.info(f1_score(predicted, test_tags_num, average=None))
        logger.info(f1_score(predicted, test_tags_num, average='weighted', pos_label=None))
    #-------------------------CRFSUITE------------------------------------
    elif data_type == "sequential":
        model_name = os.path.join(tmp_dir, 'crfsuite_model_file'+time_stamp)
        crfsuite_algorithm = config['crfsuite_algorithm'] if 'crfsuite_algorithm' in config else 'arow'
        call(['crfsuite', 'learn', '-a', crfsuite_algorithm, '-m', model_name, train_file_name])
        test_out_stream = open(test_output, 'w')
        call(['crfsuite', 'tag', '-tr', '-m', model_name, test_file_name], stdout=test_out_stream)
        test_out_stream.close()
        # parse CRFSuite output
        flattened_ref, flattened_hyp = [], []
        tag_map = {'OK': 1, 'BAD': 0}
        for line in open(test_output):
            if line == "\n":
                continue
            chunks = line.strip('\n').split('\t')
            if len(chunks) != 2:
                continue
            try:
                flattened_ref.append(tag_map[chunks[-2]])
                flattened_hyp.append(tag_map[chunks[-1]])
            except KeyError:
                continue
        print("Ref, hyp: ", len(flattened_ref), len(flattened_hyp))
        logger.info('Structured prediction f1: ')
        f1_both = f1_score(flattened_ref, flattened_hyp, average=None)
        print("F1-BAD, F1-OK: ", f1_both)
        print("F1-mult: ", f1_both[0] * f1_both[1])


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
