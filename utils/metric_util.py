from sklearn.metrics import (
    roc_auc_score, roc_curve, 
    accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score
)
from .log_util import logger
import numpy as np
from scipy import stats


def calc_metrics(y_true, y_score, threshold = 0.5):
    """ NB: return order is accuracy, f1, mcc, precision, recall
    -
    if threshold > 0: y_score = y_score > threshold; else, directly uses y_score, that's treat y_score as integer """
    if threshold > 0:
        if not isinstance(y_score, np.ndarray):
            y_score = np.array(y_score)
        y_pred_id = y_score > threshold
    else:
        y_pred_id = y_score
    accuracy = accuracy_score(y_true, y_pred_id)
    f1 = f1_score(y_true, y_pred_id)
    mcc = matthews_corrcoef(y_true, y_pred_id)
    precision = precision_score(y_true, y_pred_id)
    recall = recall_score(y_true, y_pred_id)
    return accuracy, f1, mcc, precision, recall


def calc_f1_precision_recall(y_true, y_predict):
    """  """
    # accuracy = accuracy_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    return f1, precision, recall


def find_threshold(y_true, y_score, alpha = 0.05):
    """ return threshold when fpr <= 0.05 """
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    for i, _fpr in enumerate(fpr):
        if _fpr > alpha:
            return thresh[i-1]


def roc(y_true, y_score):
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    roc = roc_auc_score(y_true, y_score)
    return roc, fpr, tpr


def calc_metrics_at_thresholds(y_true, y_pred_probability, thresholds=None, default_threshold=None):
    """  """
    if default_threshold:
        accuracy, f1, mcc, precision, recall = calc_metrics(
            y_true, y_pred_probability, default_threshold)
        logger.info(
            f'default_threshold {default_threshold}\naccuracy: {accuracy}\n'
            f"f1: {f1}\nmcc: {mcc}\nprecision: {precision}\nrecall: {recall}")

    if not thresholds: return
    for threshold in thresholds:
        accuracy, f1, mcc, precision, recall = calc_metrics(
            y_true, y_pred_probability, threshold)
        logger.info(
            f"\nthreshold {threshold}\naccuracy: {accuracy}\nf1: {f1}\nmcc: {mcc}\n"
            f"precision: {precision}\nrecall: {recall}")


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """
    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (stats.entropy(p, m) + stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def calc_spearmanr(x, y):
    """  """
    res = stats.spearmanr(x, y)
    spearman_ratio = float(res.statistic)
    return spearman_ratio