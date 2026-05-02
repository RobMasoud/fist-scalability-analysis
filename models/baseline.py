import numpy as np


def mean_fill(tensor, mask):
    """
    Simple baseline imputation: Fill missing values with the mean of observed values.
    
    Args:
        tensor: 3D tensor with missing values set to 0
        mask: binary mask (1=observed, 0=missing) with same shape as tensor
    
    Returns:
        filled_tensor: tensor with missing values filled with global mean
    """

    # copy tensor so we don't modify original
    filled_tensor = tensor.copy()

    # get observed values (where mask == 1)
    observed_values = tensor[mask == 1]

    # compute mean of observed values
    mean_value = np.mean(observed_values)

    # fill missing values (mask == 0) with mean
    filled_tensor[mask == 0] = mean_value

    return filled_tensor