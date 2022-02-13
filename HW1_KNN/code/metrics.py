import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    epsilon = 1e-100  # In order to avoid division by zero
    TP = np.sum(np.logical_and(y_pred == '1', y_true == '1'))
    TN = np.sum(np.logical_and(y_pred == '0', y_true == '0'))
    FP = np.sum(np.logical_and(y_pred == '1', y_true == '0'))
    FN = np.sum(np.logical_and(y_pred == '0', y_true == '1'))

    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    accuracy = np.sum(y_pred == y_true) / y_pred.shape[0]
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    return 1 - (np.sum(np.square(y_pred - y_true)) / (np.sum(np.square(y_pred - np.mean(y_true)))


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    return 1 / y_pred.shape[0] * np.sum(np.square(y_pred - y_true))


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    return 1 / y_pred.shape[0] * np.sum(np.abs(y_pred - y_true))
    