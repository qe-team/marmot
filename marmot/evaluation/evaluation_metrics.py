from __future__ import division
# return the f1 for (y_predicted, y_actual)

# use sklearn.metrics.f1_score with average='weighted' for evaluation
from sklearn.metrics import f1_score
import logging
import numpy as np

from marmot.experiment.import_utils import list_of_lists

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
    connections = [[min(t_end, p_end) - max(t_start, p_start) for (p_start, p_end) in pred_span] for (t_start, t_end) in true_span]
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
def sequence_correlation(y_true, y_pred, good_label=1, bad_label=0):
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
    for true_sent, pred_sent in zip(y_true, y_pred):
        assert(len(true_sent) == len(pred_sent))
        true_spans_1, true_spans_0 = get_spans(true_sent, good_label=good_label, bad_label=bad_label)
        pred_spans_1, pred_spans_0 = get_spans(pred_sent, good_label=good_label, bad_label=bad_label)

        res_1 = intersect_spans(true_spans_1, pred_spans_1)
        res_0 = intersect_spans(true_spans_0, pred_spans_0)

        sentence_pred.append((res_1+res_0)/len(true_sent))

    return sentence_pred, np.average(sentence_pred)
