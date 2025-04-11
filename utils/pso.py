import numpy as np


def fitness_function(v_pred, v_true):
    return np.sum(np.abs(v_pred - v_true) / (v_true + 1e-6))

