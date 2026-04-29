import scipy.io
import numpy as np

from preprocessing.mask_data import create_mask
from models.baseline import mean_fill
from evaluation.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

np.random.seed(42)

data = scipy.io.loadmat("data/processed/10x_data/HBA1_tensor.mat")
V = data["V"]

subs = V["subs"][0][0]
vals = V["vals"][0][0]
size = V["size"][0][0].flatten()

tensor = np.zeros(size)

for i in range(len(vals)):
    x, y, z = subs[i] - 1  # since MATLAB is 1-indexed
    tensor[x, y, z] = vals[i][0]

print("Tensor shape:", tensor.shape)

masked_tensor, mask = create_mask(tensor, missing_fraction=0.2)
predicted_tensor = mean_fill(masked_tensor, mask)

predicted_missing = predicted_tensor[mask == 0]
true_missing = tensor[mask == 0]

# only evaluate meaningful values
valid_indices = true_missing != 0
predicted_missing = predicted_missing[valid_indices]
true_missing = true_missing[valid_indices]

print("\n--- Baseline Results ---")
print("MAE:", mean_absolute_error(true_missing, predicted_missing))
print("MAPE:", mean_absolute_percentage_error(true_missing, predicted_missing))
print("R2:", r2_score(true_missing, predicted_missing))