import numpy as np


def mean_fill(tensor, mask):
    """
    Simple baseline:
    Fill missing values with the mean of observed values
    """

    # copy tensor so we don’t modify original
    filled_tensor = tensor.copy()

    # get observed values (where mask == 1)
    observed_values = tensor[mask == 1]

    # compute mean of observed values
    mean_value = np.mean(observed_values)

    # fill missing values (mask == 0) with mean
    filled_tensor[mask == 0] = mean_value

    return filled_tensor