from __future__ import print_function
#############################################################
#
#  Convert features from CRFSuite format to something else
#
############################################################
import os
import sys
import time
from argparse import ArgumentParser
from sklearn.metrics import f1_score
from subprocess import call
from marmot.util.generate_crf_template import generate_crf_template
from marmot.experiment.import_utils import mk_tmp_dir


# <in_file> -- input file
# <tmp_dir> -- directory to store the output
# <sequence> -- True - sentences as sequences, False - each word is a separate sequence
# full name of the output: "crfpp.<dataset_name>.<stamp>"
def crfsuite_to_crfpp(in_file, tmp_dir, dataset_name, sequence=True, stamp=None):
    #TODO: template
    if stamp is None:
        stamp = str(time.time())
    feature_num = 0
    out_file_name = os.path.join(tmp_dir, "crfpp." + dataset_name + '.' + stamp)
    out_file = open(out_file_name, 'w')
    tag_set = []
    for line in open(in_file):
        line = line.strip('\n').decode('utf-8')
        if line == '':
            if sequence:
                out_file.write('\n')
            continue
        elements = line.split('\t')
        cur_tag = elements[0]
        tag_set.append(cur_tag)
        cur_features = []
        for el in elements[1:]:
            stop = el.find(':')
            if stop == -1:
                cur_features.append(el)
            else:
                cur_features.append(el[:stop - 1] + el[stop + 1:])
        feature_num = len(cur_features)
        to_pr = u'\t'.join(cur_features)
        out_file.write('%s\t%s\n' % (to_pr.encode('utf-8'), cur_tag.encode('utf-8')))
        if not sequence:
            out_file.write('\n')
    generate_crf_template(feature_num, template_name='template', tmp_dir=tmp_dir)
    out_file.close()
    return out_file_name, tag_set


# <sequence> -- True - sequential representation for HMM, False - plain for classification
def crfsuite_to_svmlight(in_file, tmp_dir, dataset_name, binarized_features=None, sequence=False, stamp=None):
    if stamp is None:
        stamp = str(time.time())
    no_bin = False
    if binarized_features is None:
        print("No binary features list provided, it will be generated from the data")
        no_bin = True
        binarized_features = []

    out_file_name = os.path.join(tmp_dir, "svmlight." + dataset_name + '.' + stamp)
    out_file = open(out_file_name, 'w')
    seg_idx = 1
    tag_set = []
    tag_map = {'OK': '+1', 'BAD': '-1', u'OK': '+1', u'BAD': '-1'}
    for idx, line in enumerate(open(in_file)):
        if idx % 1000 == 0:
            sys.stderr.write('.')
        if line.strip('\n') == '':
            seg_idx += 1
            continue
        elements = line.strip('\n').decode('utf-8').split('\t')
        cur_tag = elements[0]
        tag_set.append(cur_tag)
        cur_tag_svm = tag_map[cur_tag]
        cur_features = []
        for el in elements[1:]:
            stop = el.find(':')
            cur_el = ''
            if stop == -1:
                cur_el = el
            else:
                cur_el = el[:stop] + el[stop + 1:]
            try:
                cur_features.append(binarized_features.index(cur_el) + 1)
            except:
                if no_bin:
                    binarized_features.append(cur_el)
                    cur_features.append(len(binarized_features))
        cur_features.sort()
        if sequence:
            out_file.write('%s qid:%d %s\n' % (cur_tag_svm, seg_idx, ' '.join([str(f) + ':1.0' for f in cur_features])))
        else:
            out_file.write('%s\t%s\n' % (cur_tag_svm, ' '.join([str(f) + ':1.0' for f in cur_features])))
    out_file.close()
    sys.stderr.write('\n')
    return out_file_name, tag_set, binarized_features


# <data_type> -- 'svm_light' | 'crf_suite' | 'crfpp'
def compute_ref(true_tags, out_file, data_type='svm_light'):
    tag_map = {'OK': 1, 'BAD': 0, u'OK': 1, u'BAD': 0}
    predicted = []
    if data_type == 'svm_light':
        tag_map_pred = {'+1': 1, '-1': 0}
        for line in open(out_file):
            label = line[line.find(':')+1:line.find(' ')]
            predicted.append(tag_map_pred[label])
    elif data_type == 'crfpp' or data_type == 'crf_suite':
        for line in open(out_file):
            line = line.strip('\n')
            if line == '':
                continue
            tag = line.split('\t')[-1]
            if tag == 'OK' or tag == 'BAD':
                predicted.append(tag)
        predicted = [tag_map[t] for t in predicted]
#    if (type(true_tags[0]) is str or type(true_tags[0]) is unicode) and not true_tags[0].isdigit():
    true_tags = [tag_map[t] for t in true_tags]
#    if type(predicted[0]) is str and not predicted[0].isdigit():
    print(true_tags[:10])
    print(predicted[:10])

    print(f1_score(predicted, true_tags, average=None))
    print(f1_score(predicted, true_tags, average='weighted', pos_label=None))


# extract only tags from the CRFSuite file
def get_test_tags(in_file):
    tag_set = []
    for line in open(in_file):
        line = line.strip('\n')
        if line == '':
            continue
        tag_set.append(line.split('\t')[0])
    return tag_set


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("train_file", action="store", help="path to the training features in CRFSuite format")
    parser.add_argument("test_file", action="store", help="path to the test features in CRFSuite format")
    parser.add_argument("method", help="crf_suite | crfpp | svm_light")
    parser.add_argument("representation", help="sequence | plain")
    parser.add_argument("--params", default='', help="training params, string")
    parser.add_argument("--test_params", default='', help="test params, string")
    parser.add_argument("--tmp", default=None, action="store", help="temporary directory")
    args = parser.parse_args()

    tmp_dir = args.tmp if args.tmp is not None else os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tmp_dir')
    tmp_dir = os.path.abspath(tmp_dir)
    tmp_dir = mk_tmp_dir(tmp_dir)
    stamp = args.method
    if args.params != '':
        stamp += ('.' + args.params.replace(' ', '_'))
    print("Stamp: ", stamp)
    if args.representation == 'sequence':
        sequence = True
    elif args.representation == 'plain':
        sequence = False
    else:
        print("Unknown representation: {}".format(args.representation))

    if args.method == 'crf_suite':
        model = os.path.join(tmp_dir, 'crfsuite_model_file' + stamp)
        test_tags = get_test_tags(args.test_file)
        call(['crfsuite', 'learn'] + args.params.split() + ['-m', model, args.train_file])
        test_out = open(args.test_file+'.tagged', 'w')
        call(['crfsuite', 'tag', '-tr', '-m', model, args.test_file], stdout=test_out)
        test_out.close()
        compute_ref(test_tags, args.test_file+'.tagged', data_type=args.method)
    elif args.method == 'crfpp':
        my_train_file, train_tags = crfsuite_to_crfpp(args.train_file, tmp_dir, 'train', sequence=sequence, stamp=stamp)
        my_test_file, test_tags = crfsuite_to_crfpp(args.test_file, tmp_dir, 'test', sequence=sequence, stamp=stamp)
        model = os.path.join(tmp_dir, 'crfpp_model_file' + stamp)
        print("Running training: {}".format(' '.join(['crf_learn'] + args.params.split() + [os.path.join(tmp_dir, 'template'), my_train_file, model])))
        call(['crf_learn'] + args.params.split() + [os.path.join(tmp_dir, 'template'), my_train_file, model])
        print("Running test: {}".format(' '.join(['crf_test'] + args.test_params.split() + ['-m', model, '-o', my_test_file+'.tagged', my_test_file])))
        call(['crf_test'] + args.test_params.split() + ['-m', model, '-o', my_test_file+'.tagged', my_test_file])
        compute_ref(test_tags, my_test_file+'.tagged', data_type=args.method)
    elif args.method == 'svm_light':
        my_train_file, train_tags, binarized_features = crfsuite_to_svmlight(args.train_file, tmp_dir, 'train', binarized_features=None, sequence=sequence, stamp=stamp)
        my_test_file, test_tags, binarized_features = crfsuite_to_svmlight(args.test_file, tmp_dir, 'test', binarized_features=binarized_features, sequence=sequence, stamp=stamp)
        model = os.path.join(tmp_dir, 'svm_model_file.' + stamp)
        print("Running training: {}".format(' '.join(['/export/tools/varvara/svm_multiclass/svm_light/svm_learn'] + args.params.split() + [my_train_file, model])))
        call(['/export/tools/varvara/svm_multiclass/svm_light/svm_learn'] + args.params.split() + [my_train_file, model])
        test_out = my_test_file + '.tagged'
        print("Running test: {}".format(' '.join(['/export/tools/varvara/svm_multiclass/svm_light/svm_classify', '-f', '0', my_test_file, model, test_out])))
        call(['/export/tools/varvara/svm_multiclass/svm_light/svm_classify', '-f', '0', my_test_file, model, test_out])
        compute_ref(test_tags, my_test_file+'.tagged', data_type=args.method)
    else:
        print("Unknown method: {}".format(args.method))

