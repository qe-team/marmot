# return the f1 for (y_predicted, y_actual)

# use sklearn.metrics.f1_score with average='weighted' for evaluation
from sklearn.metrics import f1_score


def weighted_fmeasure(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted', pos_label=None)
