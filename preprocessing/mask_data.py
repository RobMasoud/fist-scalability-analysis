import numpy as np


def create_mask(tensor, missing_fraction=0.2, random_seed=42):
    """
    Randomly masks a fraction of observed values in the tensor.

    Returns:
    - masked_tensor: tensor with missing values removed
    - mask: binary mask (1 = observed, 0 = missing)
    - original_values: ground truth values for evaluation
    """

    np.random.seed(random_seed)

    # flatten indices
    indices = np.arange(tensor.size)

    # shuffle indices
    np.random.shuffle(indices)

    # number of values to mask
    num_missing = int(missing_fraction * tensor.size)

    # select indices to mask
    missing_indices = indices[:num_missing]

    # create mask (start with all observed)
    mask = np.ones(tensor.size)

    # set missing positions to 0
    mask[missing_indices] = 0

    # reshape mask to match tensor
    mask = mask.reshape(tensor.shape)

    # create masked tensor
    masked_tensor = tensor.copy()
    masked_tensor[mask == 0] = 0

    return masked_tensor, mask