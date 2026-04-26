import numpy as np

from models.baseline import mean_fill
from evaluation.metrics import mean_absolute_error
from preprocessing.mask_data import create_mask


# original full data (ground truth)
tensor = np.array([
    [10, 20],
    [30, 40]
], dtype=float)

# create masked version
masked_tensor, mask = create_mask(tensor, missing_fraction=0.5)

# run baseline model
predicted_tensor = mean_fill(masked_tensor, mask)

# only evaluate where values were hidden
predicted_missing = predicted_tensor[mask == 0]
true_missing = tensor[mask == 0]

print("Original tensor:", tensor)
print("Masked tensor:", masked_tensor)
print("Mask:", mask)
print("Predicted tensor:", predicted_tensor)

print("MAE:", mean_absolute_error(true_missing, predicted_missing))