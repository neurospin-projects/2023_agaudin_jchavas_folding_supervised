# taken from https://gist.github.com/smazzanti
# with explanations from:
# https://towardsdatascience.com/how-to-calculate-roc-auc-score-for-regression-models-c0be4fdf76bb

import numpy as np


def regression_roc_auc_score(y_true, y_pred, num_rounds=10000):
    """
    Computes Regression-ROC-AUC-score.

    Parameters:
    ----------
    y_true: array-like of shape (n_samples,).
            Binary or continuous target variable.
    y_pred: array-like of shape (n_samples,). Target scores.
    num_rounds: int or string.
                If integer, number of random pairs of observations.
                If string, 'exact', all possible pairs will be evaluated.

    Returns:
    -------
    rroc: float. Regression-ROC-AUC-score.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_pairs = 0
    num_same_sign = 0

    for i, j in _yield_pairs(y_true, num_rounds):
        diff_true = y_true[i] - y_true[j]
        diff_score = y_pred[i] - y_pred[j]
        if diff_true * diff_score > 0:
            num_same_sign += 1
        elif diff_score == 0:
            num_same_sign += .5
        num_pairs += 1

    return num_same_sign / num_pairs


def _yield_pairs(y_true, num_rounds):
    """
    Returns pairs of valid indices.

    Indices must belong to observations having different values.

    Parameters:
    ----------
    y_true: array-like of shape (n_samples,).
            Binary or continuous target variable.
    num_rounds: int or string.
            If integer, number of random pairs of observations to return.
            If string, 'exact', all possible pairs will be returned.

    Yields:
    -------
    i, j: tuple of int of shape (2,). Indices referred to a pair of samples.

    """
    import numpy as np

    if num_rounds == 'exact':
        for i in range(len(y_true)):
            for j in np.where(
                    (y_true != y_true[i]) & (np.arange(len(y_true)) > i))[0]:
                yield i, j
    else:
        for r in range(num_rounds):
            i = np.random.choice(range(len(y_true)))
            j = np.random.choice(np.where(y_true != y_true[i])[0])
            yield i, j
