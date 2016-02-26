from __future__ import print_function, division

from argparse import ArgumentParser
import yaml
import logging
import copy
import sys

from marmot.experiment.import_utils import *
from marmot.experiment.preprocessing_utils import *
from marmot.experiment.learning_utils import map_classifiers, predict_all
from marmot.evaluation.evaluation_metrics import weighted_fmeasure, sequence_correlation, sequence_correlation_weighted
from marmot.evaluation.evaluation_utils import compare_vocabulary
from marmot.util.persist_features import persist_features
from marmot.evaluation.evaluation_utils import write_res_to_file

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('experiment_logger')


def label_test(flat_labels, new_test_name, text_file, method_name):
    tag_map = {0: 'BAD', 1: 'OK'}
    new_test_plain = open(new_test_name+'.'+method_name+'.plain', 'w')
    new_test_ext = open(new_test_name+'.'+method_name+'.ext', 'w')

    start_idx = 0
    for s_idx, txt in enumerate(open(text_file)):
        words = txt[:-1].decode('utf-8').strip().split()
        tag_seq = [tag_map[flat_labels[i]] for i in range(start_idx, len(words))]
        new_test_plain.write('%s\n' % ' '.join(tag_seq))
        for t_idx, (tag, word) in enumerate(zip(tag_seq, words)):
            new_test_ext.write('%s\t%d\t%d\t%s\t%s\n' % (method_name, s_idx, t_idx, word.encode('utf-8'), tag))


def main(config):
    workers = config['workers']

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
    # test
    test_data_generator = build_object(config['datasets']['test'][0])
    test_data = test_data_generator.generate()

    logger.info("Train data keys: {}".format(train_data.keys()))
    logger.info("Train data sequences: {}".format(len(train_data['target'])))
    logger.info("Sample sequence: {}".format([w.encode('utf-8') for w in train_data['target'][0]]))
#    logger.info("Sample sequence: {}".format(train_data['similarity'][0]))
#    sys.exit()

    # additional representations
    if 'representations' in config:
        representation_generators = build_objects(config['representations'])
    else:
        representation_generators = []
    for r in representation_generators:
        train_data = r.generate(train_data)
        test_data = r.generate(test_data)

    borders = config['borders'] if 'borders' in config else False

    if 'multiply_data_train' not in config:
        pass
    elif config['multiply_data_train'] == 'ngrams':
        train_data = multiply_data_ngrams(train_data, borders=borders)
    elif config['multiply_data_train'] == '1ton':
        train_data = multiply_data(train_data, borders=borders)
    elif config['multiply_data_train'] == 'duplicate':
        train_data = multiply_data_base(train_data)
    elif config['multiply_data_train'] == 'all':
        train_data = multiply_data_all(train_data, borders=borders)
    else:
        print("Unknown 'multiply data train' value: {}".format(config['multiply_data_train']))
    logger.info("Extended train representations: {}".format(len(train_data['target'])))
    #    print(train_data[:2])
    logger.info("Simple test representations: {}".format(len(test_data['target'])))
    if 'multiply_data_test' not in config:
        pass
    elif config['multiply_data_test'] == 'ngrams':
        test_data = multiply_data_ngrams(test_data, borders=borders)
    elif config['multiply_data_test'] == '1ton':
        test_data = multiply_data(test_data, borders=borders)
    else:
        print("Unknown 'multiply data test' value: {}".format(config['multiply_data_test']))
    logger.info("Extended test representations: {}".format(len(test_data['target'])))
    
    logger.info('here are the keys in your representations: {}'.format(train_data.keys()))

    # the data_type is the format corresponding to the model of the data that the user wishes to learn
    data_type = config['contexts'] if 'contexts' in config else 'plain'

    test_contexts = create_contexts(test_data, data_type=data_type)
    test_contexts_seq = create_contexts(test_data, data_type='sequential')
    train_contexts = create_contexts(train_data, data_type=data_type)

    logger.info('Vocabulary comparison -- coverage for each dataset: ')
    logger.info(compare_vocabulary([train_data['target'], test_data['target']]))
 
    # END REPRESENTATION GENERATION

    # FEATURE EXTRACTION
    train_tags = call_for_each_element(train_contexts, tags_from_contexts, data_type=data_type)
    test_tags = call_for_each_element(test_contexts, tags_from_contexts, data_type=data_type)
    test_tags_seq = call_for_each_element(test_contexts_seq, tags_from_contexts, data_type='sequential')

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

    # the way that we persist depends upon the structure of the data (plain/sequence/token_dict)
    # TODO: remove this once we have a list containing all datasets
    if config['features']['persist']:
        if 'persist_format' in config['features']:
            persist_format = config['features']['persist_format']
        else:
            persist_format = 'crf++'
        experiment_datasets = [{'name': 'test', 'features': test_features, 'tags': test_tags}, {'name': 'train', 'features': train_features, 'tags': train_tags}]
        feature_names = [f for extractor in feature_extractors for f in extractor.get_feature_names()]

        if config['features']['persist_dir']:
            persist_dir = config['features']['persist_dir']
        else:
            persist_dir = os.path.getcwd()
        logger.info('persisting your features to: {}'.format(persist_dir))
        # for each dataset, write a file and persist the features
        for dataset_obj in experiment_datasets:
            persist_features(dataset_obj['name'], dataset_obj['features'], persist_dir, feature_names=feature_names, tags=dataset_obj['tags'], file_format=persist_format)

    # BEGIN LEARNING

    # TODO: different sequence learning modules need different representation, we should wrap them in a class
    # TODO: create a consistent interface to sequence learners, will need to use *args and **kwargs because APIs are very different
    from sklearn.metrics import f1_score, precision_score, recall_score
    import numpy as np
    tag_map = {u'OK': 1, u'BAD': 0, 0: 0, 1: 1}
    if data_type == 'sequential':
        logger.info('training sequential model...')

        train_tags = [[tag_map[tag] for tag in seq] for seq in train_tags]
        test_tags = [[tag_map[tag] for tag in seq] for seq in test_tags]

        x_train = np.array([np.array(xi) for xi in train_features])
        y_train = np.array([np.array(xi) for xi in train_tags])
        x_test = np.array([numpy.array(xi) for xi in test_features])
        y_test = np.array([numpy.array(xi) for xi in test_tags])

        # pystruct
        from marmot.learning.pystruct_sequence_learner import PystructSequenceLearner
        sequence_learner = PystructSequenceLearner()
        sequence_learner.fit(x_train, y_train)
        structured_hyp = sequence_learner.predict(x_test)
        # only the last word in every sequence should be counted
        flattened_hyp = []
        flattened_ref = []
        if long_test:
            for idx, seq in enumerate(structured_hyp):
                flattened_hyp.append(seq[-1])
                flattened_ref.append(y_test[idx][-1])
        else:
            flattened_hyp = flatten(structured_hyp)
            flattened_ref = flatten(y_test)

        logger.info('scoring sequential model...')

        # TODO: the flattening is currently a hack to let us use the same evaluation code for structured and plain tasks
#        flattened_hyp = flatten(structured_hyp)

        # end pystruct
#        for idx, seq in enumerate(test_tags_seq):
#            cnt += len(seq)
#            if cnt >= len(test_predictions):
#                print("long predictions: {}, sequential: {}, sequence #{}".format(len(test_predictions), len(flatten(test_tags_seq)), idx))
#                print("Sequence: ", test_contexts_seq[idx])
#        if long_test:
#            cnt = -1
#            new_predictions = []
#            new_true = []
#            for seq in test_tags_seq:
#                cnt += len(seq)
#                new_predictions.append(tag_map[test_predictions[cnt]])
 #               new_true.append(tag_map[seq[-1]])
#            test_predictions = new_predictions
#            test_tags = new_true
#

#        print(f1_score(test_predictions, test_tags, average=None))


        print("Ref, hyp: ", len(flattened_ref), len(flattened_hyp))
        logger.info('Structured prediction f1: ')
        print(f1_score(flattened_ref, flattened_hyp, average=None))
        print(f1_score(flattened_ref, flattened_hyp, average='weighted', pos_label=None))
        logger.info("Sequence correlation: ")
        print(sequence_correlation_weighted(y_test, structured_hyp, verbose=True)[1])

    else:
        train_tags = [tag_map[tag] for tag in train_tags]
        test_tags = [tag_map[tag] for tag in test_tags]

       # data_type is 'token' or 'plain'
        logger.info('start training...')
        classifier_type = import_class(config['learning']['classifier']['module'])
        # train the classifier(s)
        classifier_map = map_classifiers(train_features, train_tags, classifier_type, data_type=data_type)
        logger.info('classifying the test instances')
        test_predictions = predict_all(test_features, classifier_map, data_type=data_type)
#        assert(len(test_predictions) == len(flatten(test_tags_seq))), "long predictions: {}, sequential: {}".format(len(test_predictions), len(flatten(test_tags_seq)))
        cnt = 0
        test_predictions_seq = []
        test_tags_seq_num = []
        tag_map = {'OK': 1, 'BAD': 0, 1: 1, 0: 0}
        long_test = True if 'multiply_data_test' in config and (config['multiply_data_test'] == 'ngrams' or config['multiply_data_test'] == '1ton') else False
        for idx, seq in enumerate(test_tags_seq):
            test_predictions_seq.append([])
            test_tags_seq_num.append([])
            for w in seq:
                test_predictions_seq[-1].append(tag_map[test_predictions[cnt]])
                test_tags_seq_num[-1].append(tag_map[w])
                cnt += 1
#            cnt += len(seq)
#            if cnt >= len(test_predictions):
#                print("long predictions: {}, sequential: {}, sequence #{}".format(len(test_predictions), len(flatten(test_tags_seq)), idx))
#                print("Sequence: ", test_contexts_seq[idx])
        if long_test:
            cnt = -1
            new_predictions = []
            new_true = []
            for seq in test_tags_seq:
                cnt += len(seq)
                new_predictions.append(tag_map[test_predictions[cnt]])
                new_true.append(tag_map[seq[-1]])
            test_predictions = new_predictions
            test_tags = new_true

        print(f1_score(test_predictions, test_tags, average=None))
        print(f1_score(test_predictions, test_tags, average='weighted', pos_label=None))
        print("Precision: {}, recall: {}".format(precision_score(test_predictions, test_tags, average=None), recall_score(test_predictions, test_tags, average=None)))
        logger.info("Sequence correlation: ")
        print(sequence_correlation_weighted(test_tags_seq_num, test_predictions_seq, verbose=True)[1])

#        label_test(test_predictions, '/export/data/varvara/marmot/marmot/experiment/final_submissions/baseline', '/export/data/varvara/corpora/wmt15_corrected/test.target', 'BASELINE')


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
