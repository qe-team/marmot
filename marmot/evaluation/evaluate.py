#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import sys, codecs
import numpy as np
from sklearn import metrics

from marmot.evaluation.evaluation_metrics import weighted_fmeasure

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('testlogger')

issues_accuracy = ['Terminology', 'Mistranslation', 'Omission', 'Addition', 'Untranslated', 'Accuracy']
issues_fluency = ['Agreement', 'Capitalization', 'Fluency', 'Function_words',
                         'Grammar', 'Morphology_(word_form)', 'Style/register',
                         'Typography', 'Unintelligible', 'Word_order',
                         'Tense/aspect/mood', 'Punctuation', 'Spelling',
                         'Part_of_speech']


def flatten(lofl):
    return [item for sublist in lofl for item in sublist]


def read_wmt_annotation(f):
    anno = {}
    for line_number, line in enumerate(f):
        line = line.decode('utf-8').strip().split()
        assert len(line) == 6, "line %d: expected 6 elements per line but found %d\n" %(line_number, len(line))
        sid, wid, w, a1, a2, a3 = line
        sid = sid.split('.')
        assert len(sid) == 2 and sid[0].isdigit() and sid[1].isdigit(), \
                "line %d: first entry (sentence id) must be if format X.Y\n" %line_number
        assert wid.isdigit(), "line %d: second entry (word index) must be integer\n" %line_number
        sid = (int(sid[0]), int(sid[1]))
        wid = int(wid)

        assert a1.lower() == "ok" or \
               a1 in issues_accuracy or \
               a1.lower() in map(str.lower, issues_accuracy) or \
               a1 in issues_fluency or \
               a1.lower() in map(str.lower, issues_fluency), \
                "line %d: unexpected error category %s\n" %(line_number, a1)
        assert a2.lower() in ['ok', 'fluency', 'accuracy'], "line %d: unexpected error category %s\n" %(line_number, a2)
        assert a3.lower() in ['ok', 'bad'], "line %d: unexpected error category %s\n" %(line_number, a3)

        if not sid in anno:
            anno[sid] = {}
        assert not wid in anno[sid], "line %d: duplicate entry for s%d:w%d" %(line_number, sid, wid)
        anno[sid][wid] = [a1.lower(), a2.lower(), a3.lower(), w]
    return anno


def generate_random_with_prior(ref_list, options):
    prior_probs = [float(ref_list.count(opt))/len(ref_list) for opt in options]
    rand_list = [options[np.random.multinomial(1, prior_probs).argmax()] for i in range(len(ref_list))]
    return rand_list


#print confusion matrix
def print_cf(cf, name, options, f1_scores, weighted_f1):
    print("----- Results for %s: -----" %name)
    print("-------------------------------------")
    print("\t\tPREDICT")
    print("REFERENCE\t", "\t".join(options))
    for linenr, line in enumerate(cf):
        print("%s\t\t" %options[linenr])
        print("\t".join(map(str,line)))
    print("-------------------------------------")
    for i in range(len(options)):
        print("F1 %24s: %f" %(options[i], f1_scores[i]))
    print("   %24s: %f" %("WEIGHTED AVG", weighted_f1))
    print("-------------------------------------")


#get scores and confusion matrix
#ref, hyp - lists of labels
def get_scores(ref, hyp, labels, name='default name', mute=0):
    assert(all([r in labels for r in ref]))
    assert(all([h in labels for h in hyp]))
    assert(len(ref) == len(hyp))

    label_list = list(labels)
    weighted_f1 = metrics.f1_score(ref, hyp, labels=label_list, average='weighted', pos_label=None)
    if not mute:
        cf_matrix = metrics.confusion_matrix(ref, hyp, labels=label_list)
        f1_scores = metrics.f1_score(ref, hyp, labels=label_list, average=None, pos_label=None)
        print_cf(cf_matrix, name, label_list, f1_scores, weighted_f1)

    return weighted_f1


#return list of labels for every example
# TODO: change the output format of the wmt parser above, this is messing everything up! - we should have dicts containing the annotation data
def choose_wmt_token_subset(anno, tok_list=None):
    #use all words
    if tok_list is None:
        return [anno[sid][wid][-2] for sid in anno for wid in anno[sid]]
    #use only words from tok_list
    else:
        # currently the index of the token in the annotation is -1, the coarse-grained annotation is at i = -2
        return [anno[sid][wid][-2] for sid in anno for wid in anno[sid] if anno[sid][wid][-1] in tok_list]


def significance_test(ref, hyp_res, options, granularity=20):
    options = list(options)
    assert type(hyp_res) != list, 'the performance on the hypothesis should be a float in the range: [0.0,1.0]'
    res_random = []
    for i in range(granularity):
        rand = generate_random_with_prior(ref, options)
        res_random.append(get_scores(ref, rand, options, str(i), mute=1))

    numerator = len([res for res in res_random if hyp_res <= res])
    if numerator == 0:
        numerator = 1

    p_value = numerator / granularity
    if p_value <= 0.05:
        print("The result is statistically significant with p = {}".format(p_value))
    else:
        print("The result is not statistically significant: {}".format(p_value))

    return p_value

# evaluate predicted and actual hashed token instances
def evaluate_hashed_predictions(ref, hyp, labels):
    ref_keys = ref.keys()
    for tok in hyp.keys():
        assert tok in ref_keys, 'The reference dict must contain the token'
        assert len(ref[tok]) == len(hyp[tok]), 'the dicts must contain the same number of instances for each token'

    label_list = set(labels)
    result_map = {}
    for tok, predicted in hyp.iteritems():
        actual = ref[tok]
        logger.info("\ttotal instances: " + str(len(predicted)))
        logger.info("Evaluating results for token = " + tok)

        hyp_res = get_scores(actual, predicted, label_list, '\''+tok+'\'')
        token_p_value = significance_test(actual, hyp_res, label_list)
        token_result = {'token': tok, 'weighted_f1': hyp_res, 'p_value': token_p_value}
        result_map[tok] = token_result

    return result_map


# assert that the keys are the same (see experiment_utils.sync)

# evaluate wmt formatted parallel files
def evaluate_wmt(anno_ref, anno_hyp, interesting_words=[]):
    option_list = ['ok', 'bad']
    # {'token': <token>, 'weighted_f1': <weighted_f1>, 'p_value': <p_value>}
    evaluation_results = {'token_level': [], 'all_data': {}}

    #scores and confusion matrices for individual words
    for tok in interesting_words:
        # choose_token_subset maps into [<tag>]
        ref_list = choose_wmt_token_subset(anno_ref, tok_list=[tok])
        hyp_list = choose_wmt_token_subset(anno_hyp, tok_list=[tok])

        hyp_res = get_scores(ref_list, hyp_list, option_list, '\''+tok+'\'')
        token_p_value = significance_test(ref_list, hyp_res, option_list)
        # {'token': <token>, 'weighted_f1': <weighted_f1>, 'p_value': <p_value>}
        token_result = {'token': tok, 'weighted_f1': hyp_res, 'p_value': token_p_value}
        evaluation_results.token_level.append(token_result)

    #scores for all interesting words or for all words if interesting_words not specified
    ref_list = choose_wmt_token_subset(anno_ref, tok_list=None)
    hyp_list = choose_wmt_token_subset(anno_hyp, tok_list=None)
    overall_result = get_scores(ref_list, hyp_list, option_list, 'all_words')
    p_value = significance_test(ref_list, overall_result, option_list)
    result_obj = {'weighted_f1': overall_result, 'p_value': p_value}
    evaluation_results['all_data'] = result_obj

    return evaluation_results


#evaluate 
def main(file_ref, file_hyp, words_file):
    ref = read_wmt_annotation(open(file_ref))
    hyp = read_wmt_annotation(open(file_hyp))
    interesting_words = [] if words_file == "" else [line[:-1].decode('utf-8') for line in open(words_file)]
    evaluate_wmt(ref, hyp, interesting_words)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ref', help="reference annotations")
    parser.add_argument('sub', help="submission annotations")
    parser.add_argument('--token_subset', help="subset of tokens to evaluate")
    args = parser.parse_args(sys.argv[1:])

    main(args.ref, args.sub, args.token_subset if args.token_subset else "")

