import numpy as np


def create_mask(tensor, missing_fraction=0.2, random_seed=42):
    """
    Randomly masks a fraction of values in the tensor.

    Returns:
    - masked_tensor: tensor with missing values set to 0
    - mask: binary mask (1 = observed, 0 = missing)
    """

    np.random.seed(random_seed)

    # create mask directly with same shape (memory efficient)
    mask = np.random.rand(*tensor.shape) > missing_fraction
    mask = mask.astype(int)

    # apply mask
    masked_tensor = tensor.copy()
    masked_tensor[mask == 0] = 0

    return masked_tensor, mask