from typing import Tuple, List
import torch
import torch.nn.functional as F
import numpy as np


def BinaryF1(prediction, target, num_classes):
    TP, TN, FP, FN = np.zeros(num_classes, dtype=int), np.zeros(num_classes, dtype=int), np.zeros(num_classes, dtype=int), np.zeros(num_classes, dtype=int)
    for cls in range(num_classes):
        TP[cls] = (prediction[target == cls] == cls).sum()
        TN[cls] = (prediction[target != cls] != cls).sum()
        FP[cls] = (prediction[target != cls] == cls).sum()
        FN[cls] = (prediction[target == cls] != cls).sum()

    # lower case metric is classwise

    accuracy = np.nan_to_num((TP + TN) / (TP + TN + FP + FN), nan=0)
    specificity = np.nan_to_num(TN / (TN + FP), nan=0)
    precision = np.nan_to_num(TP / (TP + FP), nan=0)
    recall = np.nan_to_num(TP / (TP + FN), nan=0)
    f1_score = np.nan_to_num((2 * precision * recall) / (precision + recall), nan=0)

    # uppercase metric is overall

    ACCURACY = np.nan_to_num((TP + TN).sum() / (TP + TN + FP + FN).sum(), nan=0)
    SPECIFICITY = np.nan_to_num(TN.sum() / (TN + FP).sum(), nan=0)
    PRECISION = np.nan_to_num(TP.sum() / (TP + FP).sum(), nan=0)
    RECALL = np.nan_to_num(TP.sum() / (TP + FN).sum(), nan=0)
    F1_SCORE = np.nan_to_num((2 * PRECISION * RECALL) / (PRECISION + RECALL), nan=0)

    return ACCURACY, SPECIFICITY, PRECISION, RECALL, F1_SCORE#accuracy, specificity, precision, recall, f1_score