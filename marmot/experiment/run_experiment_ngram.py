from __future__ import print_function, division

from argparse import ArgumentParser
import yaml
import logging
import os
import time
from subprocess import call

from marmot.experiment.import_utils import build_objects, build_object, call_for_each_element, import_class
from marmot.experiment.preprocessing_utils import tags_from_contexts, contexts_to_features, flatten, fit_binarizers, binarize
from marmot.experiment.context_utils import create_contexts_ngram, get_contexts_words_number
from marmot.experiment.learning_utils import map_classifiers, predict_all
from marmot.evaluation.evaluation_utils import compare_vocabulary
from marmot.util.persist_features import persist_features
from marmot.util.generate_crf_template import generate_crf_template

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


# check that everything in a data_obj matches:
#  - all source and target sentences exist
#  - alignments don't hit out of bounds
#  - target tokens really exist and are in their places
#def check_data_objects(data_obj):
#    # all representations have the same number of sentences
#    right_len = data_obj['target']
#    for a_key in data_obj:
#        assert(len(data_obj[a_key] == right_len), "Wrong number of sentences for the representation {}: expected {}, got {}".format(a_key, right_len, len(data_obj[a_key])))
#
#    # all source and target sentences exist
#    for i in range(right_len):
 #       # target sentence exists
#        assert(len(data_obj['target'][i]) > 0), "Target sentence doesn't exist for the object #{}".format(i)
#        # source sentence exists
#        assert(len(data_obj['source'][i]) > 0), "Source sentence doesn't exist for the object #{}".format(i)
#        # alignments match the target side
#        assert(len(data_obj['alignments'][i]) == len(data_obj['target'][i])), "Wrong alignments number at line #{}: expected {}, got {}".format(i, len(data_obj['target'][i]), len(data_obj['alignments'][i]))
#        # alignments match the source
#        for tok_list in data_obj['alignments'][i]:
#            for c in tok_list:
#                assert(c < len(data_obj['source'][i])), "Alignment goes out of source side bounds at line #{}: len is {}, got {}".format(i, len(data_obj['source'][i]), c)
#

def main(config):
    time_stamp = str(time.time())
    workers = config['workers']
    tmp_dir = config['tmp_dir']

    # REPRESENTATION GENERATION
    # main representations (source, target, tags)
    # training
    # many generators
#    train_data_generators = build_objects(config['datasets']['training'])
#    train_data = {}
#    for gen in train_data_generators:
#        data = gen.generate()
#        for key in data:
#            if key not in train_data:
#                train_data[key] = []
#            train_data[key].extend(data[key])
    # one generator
    train_data_generator = build_object(config['datasets']['training'][0])
    train_data = train_data_generator.generate()

    # test
    test_data_generator = build_object(config['datasets']['test'][0])
    test_data = test_data_generator.generate()

    logger.info("Train data keys: {}".format(train_data.keys()))
    logger.info("Train data sequences: {}".format(len(train_data['target'])))
    logger.info("Sample sequence: {}".format([w.encode('utf-8') for w in train_data['target'][0]]))

    # additional representations
    if 'representations' in config:
        representation_generators = build_objects(config['representations'])
    else:
        representation_generators = []
    for r in representation_generators:
        train_data = r.generate(train_data)
        test_data = r.generate(test_data)

    #check_data_objects(train_data)
    #check_data_objects(test_data)

    borders = config['borders'] if 'borders' in config else False

    logger.info('here are the keys in your representations: {}'.format(train_data.keys()))

    # the data_type is the format corresponding to the model of the data that the user wishes to learn
    data_type = config['contexts'] if 'contexts' in config else 'plain'

    test_contexts = create_contexts_ngram(test_data, data_type=data_type, test=True)
#    test_contexts_seq = create_contexts_ngram(test_data, data_type='sequential')
    print("Objects in the train data: {}".format(len(train_data['target'])))

    print("\tTraining data")
    for i in range(len(train_data['target'])):
        print("\ttarget {}, source {}, segmentation {}, source segmentation {}, alignments {}".format(len(train_data['target'][i]), len(train_data['source'][i]), len(train_data['segmentation'][i]), len(train_data['source_segmentation'][i]), len(train_data['alignments'][i])))

    train_contexts = create_contexts_ngram(train_data, data_type=data_type)
    print("Train contexts: {}".format(len(train_contexts)))
    print("1st context:", train_contexts[0])

    # the list of context objects' 'target' field lengths
    # to restore the word-level tags from the phrase-level
    #test_context_correspondence = get_contexts_words_number(test_contexts)
    test_context_correspondence = call_for_each_element(test_contexts, get_contexts_words_number, [], data_type=data_type)
    if data_type == 'sequential':
        test_context_correspondence = flatten(test_context_correspondence)

    logger.info('Vocabulary comparison -- coverage for each dataset: ')
    logger.info(compare_vocabulary([train_data['target'], test_data['target']]))
 
    # END REPRESENTATION GENERATION

    # FEATURE EXTRACTION
    train_tags = call_for_each_element(train_contexts, tags_from_contexts, data_type=data_type)
    test_tags = call_for_each_element(test_contexts, tags_from_contexts, data_type=data_type)
#    test_tags_seq = call_for_each_element(test_contexts_seq, tags_from_contexts, data_type='sequential')

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

    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.cross_validation import permutation_test_score
    import numpy as np
    tag_map = {u'OK': 1, u'BAD': 0}
    if data_type == 'sequential':
        # TODO: save features for CRFSuite, call it
        logger.info('training sequential model...')

        experiment_datasets = [{'name': 'test', 'features': test_features, 'tags': test_tags}, {'name': 'train', 'features': train_features, 'tags': train_tags}]
        feature_names = [f for extractor in feature_extractors for f in extractor.get_feature_names()]

        print("FEATURE NAMES: ", feature_names)
        persist_dir = tmp_dir
        logger.info('persisting your features to: {}'.format(persist_dir))
        # for each dataset, write a file and persist the features
        if 'persist_format' not in config:
            config['persist_format'] = 'crf_suite'
        for dataset_obj in experiment_datasets:
            persist_features(dataset_obj['name']+time_stamp, dataset_obj['features'], persist_dir, feature_names=feature_names, tags=dataset_obj['tags'], file_format=config['persist_format'])

        feature_num = len(train_features[0][0])
        train_file = os.path.join(tmp_dir, 'train'+time_stamp+'.crf')
        test_file = os.path.join(tmp_dir, 'test'+time_stamp+'.crf')

        if config['persist_format'] == 'crf++':
            # generate a template for CRF++ feature extractor
            generate_crf_template(feature_num, 'template', tmp_dir)
            # train a CRF++ model
            call(['crf_learn', os.path.join(tmp_dir, 'template'), train_file, os.path.join(tmp_dir, 'crfpp_model_file'+time_stamp)])
            # tag a test set
            call(['crf_test', '-m', os.path.join(tmp_dir, 'crfpp_model_file'+time_stamp), '-o', test_file+'.tagged', test_file])
        elif config['persist_format'] == 'crf_suite':
            crfsuite_algorithm = config['crfsuite_algorithm'] if 'crfsuite_algorithm' in config else 'arow'
            call(['crfsuite', 'learn', '-a', crfsuite_algorithm, '-m', os.path.join(tmp_dir, 'crfsuite_model_file'+time_stamp), train_file])
            test_out = open(test_file+'.tagged', 'w')
            call(['crfsuite', 'tag', '-tr', '-m', os.path.join(tmp_dir, 'crfsuite_model_file'+time_stamp), test_file], stdout=test_out)
            test_out.close()
        else:
            print("Unknown persist format: {}".format(config['persist_format']))

        sequential_true = [[]]
        sequential_predictions = [[]]
        flat_true = []
        flat_predictions = []
        for line in open(test_file+'.tagged'):
            # end of tagging, statistics reported
            if line.startswith('Performance'):
                break
            if line == '\n':
                sequential_predictions.append([])
                continue
            chunks = line[:-1].decode('utf-8').split()
            flat_true.append(chunks[-2])
            sequential_true[-1].append(chunks[-2])
            flat_predictions.append(chunks[-1])
            sequential_predictions[-1].append(chunks[-1])

        # restoring the word-level tags
        test_predictions_word, test_tags_word = [], []
        for idx, n in enumerate(test_context_correspondence):
            for i in range(n):
                test_predictions_word.append(flat_predictions[idx])
                test_tags_word.append(flat_true[idx])

        print(f1_score(test_predictions_word, test_tags_word, average=None))
        print(f1_score(test_predictions_word, test_tags_word, average='weighted', pos_label=None))
        print("Precision: {}, recall: {}".format(precision_score(test_predictions_word, test_tags_word, average=None), recall_score(test_predictions_word, test_tags_word, average=None)))

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

        # restoring the word-level tags
        test_predictions_word, test_tags_word = [], []
        for idx, n in enumerate(test_context_correspondence):
            for i in range(n):
                test_predictions_word.append(test_predictions[idx])
                test_tags_word.append(test_tags[idx])

        print(f1_score(test_tags_word, test_predictions_word, average=None))
        print(f1_score(test_tags_word, test_predictions_word, average='weighted', pos_label=None))
        print("Precision: {}, recall: {}".format(precision_score(test_predictions_word, test_tags_word, average=None), recall_score(test_predictions_word, test_tags_word, average=None)))
        # TODO: remove the hard coding of the tags here
        bad_count = sum(1 for t in test_tags if t == u'BAD' or t == 0)
        good_count = sum(1 for t in test_tags if t == u'OK' or t == 1)
        
        total = len(test_tags)
        assert (total == bad_count+good_count), 'tag counts should be correct'
        percent_good = good_count / total
        logger.info('percent good in test set: {}'.format(percent_good))
        logger.info('percent bad in test set: {}'.format(1 - percent_good))

        random_class_results = []
        random_weighted_results = []
        for i in range(20):
            random_tags_phrase = list(np.random.choice([1, 0], total, [percent_good, 1-percent_good]))
            random_tags = []
            for idx, n in enumerate(test_context_correspondence):
                for i in range(n):
                    random_tags.append(random_tags_phrase[idx])
            # random_tags = [u'GOOD' for i in range(total)]
            random_class_f1 = f1_score(test_tags_word, random_tags, average=None)
            random_class_results.append(random_class_f1)
            logger.info('two class f1 random score ({}): {}'.format(i, random_class_f1))
            # random_average_f1 = f1_score(random_tags, test_tags, average='weighted')
            random_average_f1 = f1_score(test_tags_word, random_tags, average='weighted', pos_label=None)
            random_weighted_results.append(random_average_f1)
            # logger.info('average f1 random score ({}): {}'.format(i, random_average_f1))
            
        avg_random_class = np.average(random_class_results, axis=0)
        avg_weighted = np.average(random_weighted_results)
        logger.info('two class f1 random average score: {}'.format(avg_random_class))
        logger.info('weighted f1 random average score: {}'.format(avg_weighted))


#        print("Cross-validation:")
#        print(permutation_test_score())
#        logger.info("Sequence correlation: ")
#        print(sequence_correlation_weighted(test_tags_seq_num, test_predictions_seq, verbose=True)[1])

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
