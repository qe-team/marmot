# persist features to file
# feature output formats are different depending upon the datatype
# if it's an ndarray, write to .csv
# if it's an list of lists, write to crf++ format, with a separate file containing the feature names
# if it's a dict, write to .json or pickle the object(?), write the feature names to a separate file
import os
import sys
import errno
import pandas as pd
import numpy as np
import logging

from marmot.experiment.import_utils import list_of_lists
from marmot.util.generate_crf_template import generate_crf_template

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('experiment_logger')


# write list of lists to file in CRF++ format (one item per line, empty line between sequences)
def write_lofl(lofl, filename):
    a_file = open(filename, 'w')
    for seq in lofl:
        for it in seq:
            a_file.write('%s\n' % str(it))
        a_file.write('\n')
    a_file.close()


# convert an arbitrary feature value to string
def val_to_str(f_val):
    if type(f_val) is str:
        return f_val
    elif type(f_val) is unicode:
        return f_val.encode('utf-8')
    else:
        return str(f_val)


# <word_tags> -- list of sequences of word-level tags
#    if specified - should be saved to a separate file in CRF++ format
# <phrase_lengths> -- list of phrase lengths
#    needed to be able to restore word-level tags from phrase-level
#    if specified - should be saved to a separate file in CRF++ format
#    TODO: check if matches the number of phrases
def persist_features(dataset_name, features, persist_dir, tags=None, feature_names=None, phrase_lengths=None, file_format='crf++'):
    '''
    persist the features to persist_dir -- use dataset_name as the prefix for the persisted files
    :param dataset_name: prefix of the output file
    :param features: dataset
    :param persist_dir: directory of output file(s)
    :param tags: tags for the dataset
    :param feature_names: names of features in the dataset
    :param file_format: format of the output file for sequences. Values -- 'crf++', 'crf_suite', 'svm_light'
    :return:
    '''
    try:
        os.makedirs(persist_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(persist_dir):
            pass
        else:
            raise

    if file_format == 'crf_suite' and feature_names is None:
        print("Feature names are required to save features in CRFSuite and SVMLight formats")
        return
    # for the 'plain' datatype
    if type(features) == np.ndarray and features.shape[1] == len(feature_names):
#        if file_format == 'svm_light':
#            output_path = os.path.join(persist_dir, dataset_name + '.svm')
#            output = open(output_path, 'w')
#            tags_map = {'OK': '+1', 'BAD': '-1'}
#            for a_tag, feat_seq, feat_name_seq in zip(tags, features, feature_names):
#                feat_list = [f_name + ':' + f_val for f_name, f_val in zip(feat_name_seq, feat_seq)]
#                output.write("%s %s\n" % (tags_map[a_tag], ' '.join(feat_list)))
#        else:
        output_df = pd.DataFrame(data=features, columns=feature_names)
        output_path = os.path.join(persist_dir, dataset_name + '.csv')
        output_df.to_csv(output_path, index=False)
        logger.info('saved features in: {} to file: {}'.format(dataset_name, output_path))

    # for the 'sequential' datatype
    elif list_of_lists(features):
        if file_format == 'svm_light':
            feature_names = range(1, len(features[0]) + 1)
            output_path = os.path.join(persist_dir, dataset_name + '.svm')
            output = open(output_path, 'w')
            tags_map = {'OK': '+1', 'BAD': '-1'}
            for a_tag, feat_seq in zip(tags, features):
                feat_list = []
                for f_name, f_val in zip(feature_names, feat_seq):
                    try:
                        if float(f_val) != 0.0:
                            feat_list.append(str(f_name) + ':' + val_to_str(f_val))
                    except ValueError:
                        feat_list.append(str(f_name) + ':' + val_to_str(f_val))
#                feat_list = [str(f_name) + ':' + val_to_str(f_val) for f_name, f_val in zip(feature_names, feat_seq)]
                output.write("%s %s\n" % (tags_map[a_tag], ' '.join(feat_list)))
            return
#        if file_format == 'svm_light':
#            print("SVMLight format cannot encode sequences (change the parameter 'contexts' in config from 'sequential' into 'plain')")
#            print("Example of features:", features[0])
#            return
        output_path = os.path.join(persist_dir, dataset_name + '.crf')
        output = open(output_path, 'w')
        if tags is not None:
            assert(len(features) == len(tags)), "Different numbers of tag and feature sequences"
            for s_idx, (seq, tag_seq) in enumerate(zip(features, tags)):
                assert(len(seq) == len(tag_seq)), "Lengths of tag and feature sequences don't match in sequence {}: {} and {} ({} and {})".format(s_idx, len(seq), len(tag_seq), seq, tag_seq)
                for w_idx, (feature_list, tag) in enumerate(zip(seq, tag_seq)):
#                    assert(len(feature_list) == len(feature_names)), "Wrong number of features in sequence %d, word %d: %d features, %d names" % (s_idx, w_idx, len(feature_list), len(feature_names))
                    if len(feature_list) != len(feature_names):
                        print(feature_list)
                        print(feature_names)
                        sys.exit()
                    tag = str(tag)
                    feature_str = []
                    for f in feature_list:
                        if type(f) == unicode:
                            feature_str.append(f.encode('utf-8'))
#                        else:
#                            feature_str.append(str(f))
                        else:
                            feature_str.append(f)
                    if file_format == 'crf++':
                        feature_str = '\t'.join([str(f) for f in feature_str])
                        output.write('%s\t%s\n' % (feature_str, tag))
                    elif file_format == 'crf_suite':
                        feature_str_all = []
                        for i in range(len(feature_str)):
                            if isinstance(feature_str[i], (int, float, np.float32, np.float64, np.int32, np.int64)):
                                feature_str_all.append(feature_names[i] + '=1:' + str(feature_str[i]))
                            else:
                                feature_str_all.append(feature_names[i] + '=' + str(feature_str[i]))
#                        feature_str = [feature_names[i] + '=' + feature_str[i] for i in range(len(feature_str))]
                        feature_str = '\t'.join(feature_str_all)
                        output.write("%s\t%s\n" % (tag, feature_str))
                    else:
                        print("Unknown data format:", file_format)
                        return False
                output.write("\n")
        else:
            for s_idx, seq in enumerate(features):
                for w_idx, feature_list in enumerate(seq):
                    #assert(len(seq) == len(feature_names)), "Wrong number of features in sequence %d, word %d" % (s_idx, w_idx)
                    feature_str = []
                    for f in feature_list:
                        if type(f) == unicode:
                            feature_str.append(f.encode('utf-8'))
#                        else:
#                            feature_str.append(str(f))
                        else:
                            feature_str.append(f)
                    if file_format == 'crf++':
                        feature_str = '\t'.join([str(f) for f in feature_str])
                    elif file_format == 'crf_suite':
#                        feature_str = [feature_names[i] + '=' + feature_str[i] for i in range(len(feature_str))]
                        feature_str_all = []
                        for i in range(len(feature_str)):
                            if isinstance(feature_str[i], (int, float, np.float32, np.float64, np.int32, np.int64)):
                                feature_str_all.append(feature_names[i] + '=1:' + str(feature_str[i]))
                            else:
                                feature_str_all.append(feature_names[i] + '=' + str(feature_str[i]))
                        feature_str = '\t'.join(feature_str_all)
                    else:
                        print("Unknown data format:", file_format)
                        return False
                    output.write("%s\n" % feature_str)
                output.write("\n")
        if feature_names is not None:
            output_features = open(os.path.join(persist_dir, dataset_name + '.features'), 'w')
            for f_name in feature_names:
                output_features.write("%s\n" % f_name.encode('utf-8'))
            output_features.close()
        output.close()

        # write phrase lengths
        if phrase_lengths is not None:
            write_lofl(phrase_lengths, os.path.join(persist_dir, dataset_name + '.phrase-lengths'))

        # generate CRF++ template
        if file_format == 'crf++':
            feature_num = len(features[0][0])
            generate_crf_template(feature_num, tmp_dir=persist_dir)
