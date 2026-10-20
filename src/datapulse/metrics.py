"""Evaluation helpers that emphasize ranking and actionable classification."""

import numpy as np
from sklearn.metrics import precision_recall_curve


def threshold_for_recall(y_true, probabilities, minimum_recall=0.70):
    """Return the highest threshold meeting a minimum positive-class recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    eligible = np.flatnonzero(recall[:-1] >= minimum_recall)
    if not len(eligible):
        return float(thresholds[0]) if len(thresholds) else 0.5
    return float(thresholds[eligible[-1]])

