from __future__ import print_function, division

from argparse import ArgumentParser
import yaml
import logging
import sys
import os
from subprocess import call
from sklearn.metrics import f1_score

from marmot.experiment.import_utils import call_for_each_element, build_object, build_objects, mk_tmp_dir
from marmot.experiment.preprocessing_utils import create_contexts, tags_from_contexts, contexts_to_features, fit_binarizers, binarize, flatten
from marmot.evaluation.evaluation_utils import compare_vocabulary
from marmot.util.persist_features import persist_features
from marmot.util.generate_crf_template import generate_crf_template

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('experiment_logger')
'''
Only feature extraction
Extract features and save in CRF++, CRFSuite or SVMLight format
'''


def feat_to_string(a_feat):
    try:
        return a_feat.encode('utf-8')
    except:
        return str(a_feat)


# data type - plain
def binarize_features(train_features, feature_names, train_tags):
    binary_features = set()
    for features, a_tag in zip(train_features, train_tags):
        for a_feat, a_name in zip(features, feature_names):
            new_feature = "{}_{}_{}".format(a_name, feat_to_string(a_feat), a_tag)
            binary_features.add(new_feature)
    return list(binary_features)


# data type - plain
# no tag in the feature
def binarize_features_blind(train_features, feature_names):
    binary_features = set()
    for features in train_features:
        for a_feat, a_name in zip(features, feature_names):
            new_feature = "{}_{}".format(a_name, feat_to_string(a_feat))
            binary_features.add(new_feature)
    return list(binary_features)


# features, tags -- dataset to binarize
# feature_names -- feature names for this dataset
# binary_features -- list of binary feature names
# output -- list of binary feature names which light for this object
def get_binary_features(test_features, feature_names, test_tags, binary_features):
    new_test_features = []
    for features, a_tag in zip(test_features, test_tags):
        cur_features = []
        for a_feat, a_name in zip(features, feature_names):
            try:
                cur_features.append(binary_features.index('{}_{}_{}'.format(a_name, feat_to_string(a_feat), a_tag)) + 1)
            except ValueError:  # no such feature, skipping
                pass
        cur_features.sort()
        # print("Features: ", cur_features)
        new_test_features.append(cur_features)
    return new_test_features


# the same, but features without tag
def get_binary_features_blind(test_features, feature_names, binary_features):
    new_test_features = []
    for features in test_features:
        cur_features = []
        for a_feat, a_name in zip(features, feature_names):
            try:
                cur_features.append(binary_features.index('{}_{}'.format(a_name, feat_to_string(a_feat))) + 1)
            except ValueError:  # no such feature, skipping
                pass
        cur_features.sort()
        # print("Features: ", cur_features)
        new_test_features.append(cur_features)
    return new_test_features


# binary features for test
#two variants for every object: with positive and negative features
def get_binary_features_test(test_features, feature_names, test_tags, binary_features):
    new_test_features = []
    new_test_features_inverse = []
    opposite = {'OK': 'BAD', 'BAD': 'OK'}
    for features, a_tag in zip(test_features, test_tags):
        cur_features_dir = []
        cur_features_inv = []
        for a_feat, a_name in zip(features, feature_names):
            try:
                cur_features_dir.append(binary_features.index('{}_{}_{}'.format(a_name, feat_to_string(a_feat), a_tag)) + 1)
                cur_features_inv.append(binary_features.index('{}_{}_{}'.format(a_name, feat_to_string(a_feat), opposite[a_tag])) + 1)
            except ValueError:  # no such feature, skipping
                pass
        cur_features_dir.sort()
        cur_features_inv.sort()
        # print("Features: ", cur_features)
        new_test_features.append(cur_features_dir)
        new_test_features_inverse.append(cur_features_inv)
    return new_test_features, new_test_features_inverse


def get_test_score(test_file, inverse_test_file):
    predicted = []
    tag_map = {'+1': 1, '-1': 0}
    dir_score, inv_score = 0.0, 0.0
    for line_dir, line_inv in zip(open(test_file), open(inverse_test_file)):
        dir_label = line_dir[line_dir.find(':')+1:line_dir.find(' ')]
        inv_label = line_inv[line_inv.find(':')+1:line_inv.find(' ')]
        dir_score = line_dir[:line_dir.find(':')]
        inv_score = line_inv[:line_inv.find(':')]
        if dir_score > inv_score:
            predicted.append(tag_map[dir_label])
        else:
            predicted.append(tag_map[inv_label])
    return predicted


# parse SVLight output,
# return the predicted tags (0 - BAD, 1 - GOOD)
def get_test_score_blind(test_file):
    predicted = []
    tag_map = {'+1': 1, '-1': 0}
    for line in open(test_file):
        label = line[line.find(':')+1:line.find(' ')]
        predicted.append(tag_map[label])
    return predicted


# persist features to svm_light format
# all features - binary
# feature = <feature_name>_<feature_value>_<label>
def persist_to_svm(train_features, test_features, feature_names, train_tags, test_tags, persist_dir):
    # binarize
    logger.info("Binarize features")
    binary_features = binarize_features_blind(train_features, feature_names, train_tags)
    logger.info("Get binary representation for test")
    new_test_features = get_binary_features(test_features, feature_names, test_tags, binary_features)
    test_file_name = os.path.join(persist_dir, 'test_binary.svm')
    test_file = open(test_file_name, 'w')
    tags_map = {'OK': '+1', 'BAD': '-1'}
    for feat, a_tag in zip(new_test_features, test_tags):
        #print("Features: ", feat)
        #print("Tag: ", a_tag)
        test_file.write('%s %s\n' % (tags_map[a_tag], ' '.join([str(f) + ':1.0' for f in feat])))

    logger.info("Get binary representation for training")
    new_train_features = get_binary_features(train_features, feature_names, train_tags, binary_features)

    # persist
    logger.info("Export training and test")
    train_file_name = os.path.join(persist_dir, 'train_binary.svm')
    #test_file_name = os.path.join(persist_dir, 'test_binary.svm')
    train_file = open(train_file_name, 'w')
    #test_file = open(test_file_name, 'w')
    #tags_map = {'OK': '+1', 'BAD': '-1'}
    for feat, a_tag in zip(new_train_features, train_tags):
#        print("Features: ", feat)
#        print("Tag: ", a_tag)
        train_file.write('%s %s\n' % (tags_map[a_tag], ' '.join([str(f) + ':1.0' for f in feat])))
    #for feat, a_tag in zip(new_test_features, test_tags):
    #    test_file.write('%s %s\n' % (tags_map[a_tag], ' '.join([str(f) + ':1.0' for f in feat])))
    train_file.close()
    test_file.close()
    # persist unbinarized
#    logger.info("Export non-binary versions for control")
#    train_control = open(os.path.join(persist_dir, 'train_control.svm'), 'w')
#    test_control = open(os.path.join(persist_dir, 'test_control.svm'), 'w')
#    for feat, a_tag in zip(train_features, train_tags):
#        train_control.write("%s %s\n" % (a_tag, ' '.join([str(f_name) + ':' + feat_to_string(f) for f_name, f in zip(feature_names, feat)])))
#    for feat, a_tag in zip(test_features, test_tags):
#        test_control.write("%s %s\n" % (a_tag, ' '.join([str(f_name) + ':' + feat_to_string(f) for f_name, f in zip(feature_names, feat)])))
#    train_control.close()
#    test_control.close()
    return train_file_name, test_file_name


# persist to svm with double test file
def persist_to_svm_dbl(train_features, test_features, feature_names, train_tags, test_tags, persist_dir):
    # binarize
    logger.info("Binarize features")
    binary_features = binarize_features(train_features, feature_names, train_tags)
    logger.info("Get binary representation for test")
    new_test_features_dir, new_test_features_inv = get_binary_features_test(test_features, feature_names, test_tags, binary_features)

    test_file_name = os.path.join(persist_dir, 'test_binary_dir.svm')
    test_file = open(test_file_name, 'w')
    tags_map = {'OK': '+1', 'BAD': '-1'}
    for feat, a_tag in zip(new_test_features_dir, test_tags):
        test_file.write('%s %s\n' % (tags_map[a_tag], ' '.join([str(f) + ':1.0' for f in feat])))

    inv_test_file_name = os.path.join(persist_dir, 'test_binary_inv.svm')
    inv_test_file = open(inv_test_file_name, 'w')
    tags_map_inv = {'OK': '-1', 'BAD': '+1'}
    for feat, a_tag in zip(new_test_features_inv, test_tags):
        inv_test_file.write('%s %s\n' % (tags_map_inv[a_tag], ' '.join([str(f) + ':1.0' for f in feat])))

    logger.info("Get binary representation for training")
    new_train_features = get_binary_features(train_features, feature_names, train_tags, binary_features)

    # persist
    logger.info("Export training and test")
    train_file_name = os.path.join(persist_dir, 'train_binary.svm')
    #test_file_name = os.path.join(persist_dir, 'test_binary.svm')
    train_file = open(train_file_name, 'w')
    for feat, a_tag in zip(new_train_features, train_tags):
        train_file.write('%s %s\n' % (tags_map[a_tag], ' '.join([str(f) + ':1.0' for f in feat])))
    train_file.close()
    test_file.close()
    inv_test_file.close()
    return train_file_name, test_file_name, inv_test_file_name


# persist to svm without tag encoded in features
def persist_to_svm_blind(train_features, test_features, train_tags, test_tags, feature_names, persist_dir):
    # binarize
    logger.info("Binarize features")
    binary_features = binarize_features_blind(train_features, feature_names)
    logger.info("Get binary representation for test")
    new_test_features = get_binary_features_blind(test_features, feature_names, binary_features)

    test_file_name = os.path.join(persist_dir, 'test_binary.svm')
    test_file = open(test_file_name, 'w')
    tags_map = {'OK': '+1', 'BAD': '-1'}
    for feat, a_tag in zip(new_test_features, test_tags):
        test_file.write('%s %s\n' % (tags_map[a_tag], ' '.join([str(f) + ':1.0' for f in feat])))

    logger.info("Get binary representation for training")
    new_train_features = get_binary_features_blind(train_features, feature_names, binary_features)

    # persist
    logger.info("Export training and test")
    train_file_name = os.path.join(persist_dir, 'train_binary.svm')
    #test_file_name = os.path.join(persist_dir, 'test_binary.svm')
    train_file = open(train_file_name, 'w')
    for feat, a_tag in zip(new_train_features, train_tags):
        train_file.write('%s %s\n' % (tags_map[a_tag], ' '.join([str(f) + ':1.0' for f in feat])))
    train_file.close()
    test_file.close()
    return train_file_name, test_file_name


def main(config):
    workers = config['workers']
    tmp_dir = config['tmp_dir']
    tmp_dir = mk_tmp_dir(tmp_dir)

    # REPRESENTATION GENERATION
    # main representations (source, target, tags)
    # training
    train_data_generators = build_objects(config['datasets']['training'])
    train_data = {}
    for gen in train_data_generators:
        data = gen.generate()
        for key in data:
            if key not in train_data:
                train_data[key] = []
            train_data[key].extend(data[key])
    dev, test = False, False
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
    data_type = config['contexts']
    print("DATA TYPE:", data_type)
#    sys.exit()

    train_contexts = create_contexts(train_data, data_type=data_type)
    if test:
        test_contexts = create_contexts(test_data, data_type=data_type)
    if dev:
        dev_contexts = create_contexts(dev_data, data_type=data_type)

    logger.info('Vocabulary comparison -- coverage for each dataset: ')
    logger.info(compare_vocabulary([train_data['target'], test_data['target']]))

    # END REPRESENTATION GENERATION

    # FEATURE EXTRACTION
    train_tags = call_for_each_element(train_contexts, tags_from_contexts, data_type=data_type)
    if test:
        test_tags = call_for_each_element(test_contexts, tags_from_contexts, data_type=data_type)
    if dev:
        dev_tags = call_for_each_element(dev_contexts, tags_from_contexts, data_type=data_type)

    logger.info('creating feature extractors...')
    feature_extractors = build_objects(config['feature_extractors'])
    if test:
        logger.info('mapping the feature extractors over the contexts for test...')
        test_features = call_for_each_element(test_contexts, contexts_to_features, [feature_extractors, workers], data_type=data_type)
        print("Test features sample: ", test_features[0])
    if dev:
        logger.info('mapping the feature extractors over the contexts for dev...')
        dev_features = call_for_each_element(dev_contexts, contexts_to_features, [feature_extractors, workers], data_type=data_type)
    logger.info('mapping the feature extractors over the contexts for train...')
    train_features = call_for_each_element(train_contexts, contexts_to_features, [feature_extractors, 1], data_type=data_type)
    print("Train features sample: ", train_features[0])

    logger.info('number of training instances: {}'.format(len(train_features)))
    logger.info('number of testing instances: {}'.format(len(test_features)))

    logger.info('All of your features now exist in their raw representation, but they may not be numbers yet')
    # END FEATURE EXTRACTION

    # binarizing features
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

        feature_names = [f for extractor in feature_extractors for f in extractor.get_feature_names()]
        features_num = len(feature_names)
        true_features_num = len(all_values[0])

        logger.info('fitting binarizers...')
        binarizers = fit_binarizers(all_values)
        logger.info('binarizing test data...')
        test_features = call_for_each_element(test_features, binarize, [binarizers], data_type=data_type)
        logger.info('binarizing training data...')
        # TODO: this line hangs with alignment+w2v
        train_features = call_for_each_element(train_features, binarize, [binarizers], data_type=data_type)

        logger.info('All of your features are now scalars in numpy arrays')
        logger.info('training and test sets successfully generated')

    # persisting features
    logger.info('training and test sets successfully generated')

#    experiment_datasets = [{'name': 'train', 'features': train_features, 'tags': train_tags}]
#    if test:
#        experiment_datasets.append({'name': 'test', 'features': test_features, 'tags': test_tags})
#    if dev:
#        experiment_datasets.append({'name': 'dev', 'features': dev_features, 'tags': dev_tags})
#    feature_names = [f for extractor in feature_extractors for f in extractor.get_feature_names()]

    feature_names = [f for extractor in feature_extractors for f in extractor.get_feature_names()]
    persist_dir = config['persist_dir'] if 'persist_dir' in config else config['features']['persist_dir']
    persist_dir = mk_tmp_dir(persist_dir)
#    train_file_name, test_file_name, inv_test_file_name = persist_to_svm_dbl(train_features, test_features, feature_names, train_tags, test_tags, persist_dir)
    train_file_name, test_file_name = persist_to_svm_blind(train_features, test_features, train_tags, test_tags, feature_names, persist_dir)
    model_name = os.path.join(persist_dir, 'model')
    logger.info("Start training")
    kernel = 0  # linear kernel (default)
    if 'svm_params' in config:
        kernel = int(config['svm_params']['kernel']) if kernel <= 4 else 0
    call(['/export/tools/varvara/svm_multiclass/svm_light/svm_learn', '-t', str(kernel), train_file_name, model_name])
    logger.info("Training completed, start testing")
    test_file = os.path.join(persist_dir, 'out')
#    inverse_test_file = os.path.join(persist_dir, 'out_inv')
    call(['/export/tools/varvara/svm_multiclass/svm_light/svm_classify', '-f', '0', test_file_name, model_name, test_file])
#    call(['/export/tools/varvara/svm_multiclass/svm_light/svm_classify', '-f', '0', inv_test_file_name, model_name, inverse_test_file])
    logger.info("Testing completed")
#    predicted = get_test_score(test_file, inverse_test_file)
    predicted = get_test_score_blind(test_file)
    tag_map = {'OK': 1, 'BAD': 0}
    test_tags_num = [tag_map[t] for t in test_tags]
    logger.info(f1_score(predicted, test_tags_num, average=None))
    logger.info(f1_score(predicted, test_tags_num, average='weighted', pos_label=None))

#    persist_format = config['persist_format'] if 'persist_format' in config else config['features']['persist_format']
#    logger.info('persisting your features to: {}'.format(persist_dir))
#    # for each dataset, write a file and persist the features
#    for dataset_obj in experiment_datasets:
#        persist_features(dataset_obj['name'], dataset_obj['features'], persist_dir, feature_names=feature_names, tags=dataset_obj['tags'], file_format=persist_format)
#    # generate a template for CRF++ feature extractor
#    feature_num = len(feature_names)
#    if persist_format == 'crf++':
#        generate_crf_template(feature_num, 'template', persist_dir)

#    logger.info('Features persisted to: {}'.format(', '.join([os.path.join(persist_dir, nn) for nn in [obj['name'] for obj in experiment_datasets]])))


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("configuration_file", action="store", help="path to the config file (in YAML format).")
    parser.add_argument("--tmp", action="store", default=None, help="temporary directory")
    args = parser.parse_args()
    experiment_config = {}

    # Experiment hyperparams
    cfg_path = args.configuration_file
    # read configuration file
    with open(cfg_path, "r") as cfg_file:
        experiment_config = yaml.load(cfg_file.read())
    if args.tmp is not None:
        experiment_config['tmp_dir'] = args.tmp
    main(experiment_config)
