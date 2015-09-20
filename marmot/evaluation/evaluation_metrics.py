from __future__ import division
# return the f1 for (y_predicted, y_actual)

# use sklearn.metrics.f1_score with average='weighted' for evaluation
from sklearn.metrics import f1_score, accuracy_score
import logging
import numpy as np

from marmot.experiment.import_utils import list_of_lists
from marmot.experiment.preprocessing_utils import flatten

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('experiment_logger')


def weighted_fmeasure(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted', pos_label=None)


# each span is a pair (span start, span end). span end = last span element + 1
def get_spans(sentence, good_label=1, bad_label=0):
    good_spans, bad_spans = [], []
    prev_label = None
    cur_start = 0
    for idx, label in enumerate(sentence):
        if label == good_label:
            if label != prev_label:
                if prev_label is not None:
                    bad_spans.append((cur_start, idx))
                cur_start = idx
        elif label == bad_label:
            if label != prev_label:
                if prev_label is not None:
                    good_spans.append((cur_start, idx))
                cur_start = idx
        else:
            print("Unknown label", label)
        prev_label = label
    # add last span
    if prev_label == good_label:
        good_spans.append((cur_start, len(sentence)))
    else:
        bad_spans.append((cur_start, len(sentence)))
    return(good_spans, bad_spans)


def intersect_spans(true_span, pred_span):
    # connectivity matrix for all pairs of spans from the reference and prediction
    connections = [[max(0, min(t_end, p_end) - max(t_start, p_start)) for (p_start, p_end) in pred_span] for (t_start, t_end) in true_span]
    adjacency = np.array(connections)
    res = 0
    # while there are non-zero elements == there are unused spans
    while adjacency.any():
        # maximum intersection
        max_el = adjacency.max()
        max_coord = adjacency.argmax()
        # coordinates of the max element
        coord_x, coord_y = max_coord // adjacency.shape[1], max_coord % adjacency.shape[1]
        res += max_el

        # remove all conflicting edges
        for i in range(adjacency.shape[0]):
            adjacency[i][coord_y] = 0
        for i in range(adjacency.shape[1]):
            adjacency[coord_x][i] = 0

    return res


# Y_true and y_pred - lists of sequences
#def sequence_correlation(y_true, y_pred, good_label=1, bad_label=0):
#    assert(len(y_true) == len(y_pred))
#    if not list_of_lists(y_true) and not list_of_lists(y_pred):
#        logger.warning("You provided the labels in a flat list of length {}. Assuming them to be one sequence".format(len(y_true)))
#        y_true = [y_true]
#        y_pred = [y_pred]
#    elif list_of_lists(y_true) and list_of_lists(y_pred):
#        pass
#    else:
#        logger.error("Shapes of the hypothesis and the reference don't match")
#        return 0
#
#    sentence_pred = []
#    for true_sent, pred_sent in zip(y_true, y_pred):
#        assert(len(true_sent) == len(pred_sent))
#        true_spans_1, true_spans_0 = get_spans(true_sent, good_label=good_label, bad_label=bad_label)
#        pred_spans_1, pred_spans_0 = get_spans(pred_sent, good_label=good_label, bad_label=bad_label)
#
#        res_1 = intersect_spans(true_spans_1, pred_spans_1)
#        res_0 = intersect_spans(true_spans_0, pred_spans_0)
#
#        sentence_pred.append((res_1+res_0)/len(true_sent))
#
#    return sentence_pred, np.average(sentence_pred)

# Y_true and y_pred - lists of sequences
def sequence_correlation(y_true, y_pred, good_label=1, bad_label=0, out='sequence_corr.out', verbose=False):
    assert(len(y_true) == len(y_pred))
    if not list_of_lists(y_true) and not list_of_lists(y_pred):
        logger.warning("You provided the labels in a flat list of length {}. Assuming them to be one sequence".format(len(y_true)))
        y_true = [y_true]
        y_pred = [y_pred]
    elif list_of_lists(y_true) and list_of_lists(y_pred):
        pass
    else:
        logger.error("Shapes of the hypothesis and the reference don't match")
        return 0

    sentence_pred = []
    if verbose:
        out_file = open(out, 'w')
    for true_sent, pred_sent in zip(y_true, y_pred):
        assert(len(true_sent) == len(pred_sent))
        true_spans_1, true_spans_0 = get_spans(true_sent, good_label=good_label, bad_label=bad_label)
        pred_spans_1, pred_spans_0 = get_spans(pred_sent, good_label=good_label, bad_label=bad_label)

        res_1 = intersect_spans(true_spans_1, pred_spans_1)
        res_0 = intersect_spans(true_spans_0, pred_spans_0)

        corr_val = (res_1+res_0)/float(len(true_sent))
#        print(corr_val, type(corr_val))
        if verbose:
            out_file.write("Reference:  %s\nPrediction: %s\nCorrelation: %s\n" % (' '.join([str(t) for t in true_sent]), ' '.join([str(t) for t in pred_sent]), str(corr_val)))
        sentence_pred.append(corr_val)

    if verbose:
        out_file.close()
    return sentence_pred, np.average(sentence_pred)


def sequence_correlation_weighted(y_true, y_pred, good_label=1, bad_label=0, out='sequence_corr.out', verbose=False):
    assert(len(y_true) == len(y_pred))
    if not list_of_lists(y_true) and not list_of_lists(y_pred):
        logger.warning("You provided the labels in a flat list of length {}. Assuming them to be one sequence".format(len(y_true)))
        y_true = [y_true]
        y_pred = [y_pred]
    elif list_of_lists(y_true) and list_of_lists(y_pred):
        pass
    else:
        logger.error("Shapes of the hypothesis and the reference don't match")
        return 0

    sentence_pred = []
    if verbose:
        out_file = open(out, 'w')
    for true_sent, pred_sent in zip(y_true, y_pred):
        ref_bad = sum([1 for l in true_sent if l == bad_label])
        ref_good = sum([1 for l in true_sent if l == good_label])
        assert(ref_bad + ref_good == len(true_sent))
        # coefficients that ensure the equal influence of good and bad classes on the overall score
        try:
            coeff_bad = len(true_sent)/(2*ref_bad)
        except ZeroDivisionError:
            coeff_bad = 0.0
        try:
            coeff_good = len(true_sent)/(2*ref_good)
        except ZeroDivisionError:
            coeff_good = 0.0

        assert(len(true_sent) == len(pred_sent))
        true_spans_1, true_spans_0 = get_spans(true_sent, good_label=good_label, bad_label=bad_label)
        pred_spans_1, pred_spans_0 = get_spans(pred_sent, good_label=good_label, bad_label=bad_label)

        res_1 = intersect_spans(true_spans_1, pred_spans_1)
        res_0 = intersect_spans(true_spans_0, pred_spans_0)

        len_t_1, len_t_0 = len(true_spans_1), len(true_spans_0)
        len_p_1, len_p_0 = len(pred_spans_1), len(pred_spans_0)
        if len_t_1 + len_t_0 > len_p_1 + len_p_0:
            spans_ratio = (len_p_1 + len_p_0)/(len_t_1 + len_t_0)
        else:
            spans_ratio = (len_t_1 + len_t_0)/(len_p_1 + len_p_0)

        corr_val = (res_1*coeff_good + res_0*coeff_bad)*spans_ratio/float(len(true_sent))
#        try:
#            corr_val = res_0/float(ref_bad)
#        except ZeroDivisionError:
#            corr_val = 1.0
#        print(corr_val, type(corr_val))
        if verbose:
            out_file.write("Reference:  %s\nPrediction: %s\nCorrelation: %s\n" % (' '.join([str(t) for t in true_sent]), ' '.join([str(t) for t in pred_sent]), str(corr_val)))
        sentence_pred.append(corr_val)

    if verbose:
        out_file.close()
    return sentence_pred, np.average(sentence_pred)


# sequence correlation based on full (not restricted) accuracy score
# accuracy score weighted by the importance of tags times ratio of numbers of spans in the hypothesis and the reference
def sequence_correlation_simple(true_tags, test_tags):
    seq_corr_all = []
    for true_seq, test_seq in zip(true_tags, test_tags):
        n_spans_1_true, n_spans_0_true = 0, 0
        n_spans_pred = 0
        prev_true = None
        for tag in true_seq:
            if tag == 1 and prev_true == 0:
                n_spans_0_true += 1
            elif tag == 0 and prev_true == 1:
                n_spans_1_true += 1
            prev_true = tag
        if true_seq[-1] == 0:
            n_spans_0_true += 1
        elif true_seq[-1] == 1:
            n_spans_1_true += 1
        prev_pred = None
        for tag in test_seq:
            if tag != prev_pred:
                n_spans_pred += 1
            prev_pred = tag
        n_spans_pred -= 1
        lambda_0 = len(test_tags)/n_spans_0_true if n_spans_0_true != 0 else 0
        lambda_1 = len(test_tags)/n_spans_1_true if n_spans_1_true != 0 else 0
        weights = []
        for t in true_seq:
            if t == 1:
                weights.append(lambda_1)
            elif t == 0:
                weights.append(lambda_0)
            else:
                print("Unknown reference tag: {}".format(t))
        assert(len(weights) == len(true_seq)), "Expected weights array len {}, got {}".format(len(weights), len(true_tags))
        acc = accuracy_score(true_seq, test_seq, sample_weight=weights)
        # penalises any difference in the number of spans between the reference and the hypothesis
        n_spans_true = n_spans_1_true + n_spans_0_true - 1
        if n_spans_true == 0 and n_spans_pred == 0:
            seq_corr_all.append(1)
        else:
            if n_spans_true == 0 or n_spans_pred == 0:
                seq_corr_all.append(0)
            else:
                ratio = min(n_spans_pred/n_spans_true, n_spans_true/n_spans_pred)
                seq_corr_all.append(acc*ratio)
    return seq_corr_all, np.average(seq_corr_all)


def cohens_kappa(true_tags, test_tags, verbose=False):
    # true positive, true negative, false positive, false negative
    tp, tn, fp, fn = 0, 0, 0, 0
    flat_true = flatten(true_tags)
    flat_test = flatten(test_tags)
    n_tags = len(flat_true)
    for true, test in zip(flat_true, flat_test):
        if true == 0:
            if test == 0:
                tn += 1
            elif test == 1:
                fp += 1
        elif true == 1:
            if test == 0:
                fn += 1
            elif test == 1:
                tp += 1
    # observed agreement, expected agreement
    po = (tp + tn)/n_tags
    prob_1_true = (tp + fn)/n_tags
    prob_1_test = (tp + fp)/n_tags
    pe = prob_1_true*prob_1_test + (1 - prob_1_true)*(1 - prob_1_test)
    if verbose:
        print('''
                 Pred   |          |
                        |    OK    |   BAD     
              True      |          |       
              ---------------------------
                OK      |   %d  |   %d
              ---------------------------
                BAD     |   %d  |   %d
              ---------------------------
              ''' % (tp, fn, fp, tn))
        print("Tp %d, fp %d, tn %d, fn %d" % (tp, fp, tn, fn))
    return (po - pe)/(1 - pe)
