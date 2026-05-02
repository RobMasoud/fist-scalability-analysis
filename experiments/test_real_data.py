"""
Main experiment script for testing imputation methods on real spatial transcriptomics data.

Loads HBA1 dataset, creates 20% missing data mask, runs baseline/REMAP/ZIFA methods,
and evaluates performance using MAE/MAPE/R² on non-zero values.
"""

import scipy.io
import numpy as np

from preprocessing.mask_data import create_mask
from models.baseline import mean_fill
from models.zifa_model import zifa_impute
from models.remap_model import remap_impute_tensor, build_remap_args
from evaluation.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

np.random.seed(42)

# Load sparse tensor data from MATLAB format
data = scipy.io.loadmat("data/processed/10x_data/HBA1_tensor.mat")
V = data["V"]

subs = V["subs"][0][0]
vals = V["vals"][0][0]
size = V["size"][0][0].flatten()

# Reconstruct dense tensor from sparse representation
tensor = np.zeros(size)

for i in range(len(vals)):
    x, y, z = subs[i] - 1  # since MATLAB is 1-indexed
    tensor[x, y, z] = vals[i][0]

print("Tensor shape:", tensor.shape)

# Create 20% missing data mask
masked_tensor, mask = create_mask(tensor, missing_fraction=0.2)

# Run baseline imputation (mean fill)
predicted_tensor = mean_fill(masked_tensor, mask)

# Extract predictions and true values for missing entries
predicted_missing = predicted_tensor[mask == 0]
true_missing = tensor[mask == 0]

# Only evaluate on non-zero true values (avoid division by zero in MAPE)
valid_indices = true_missing != 0
predicted_missing = predicted_missing[valid_indices]
true_missing = true_missing[valid_indices]

print("\n--- Baseline Results ---")
print("MAE:", mean_absolute_error(true_missing, predicted_missing))
print("MAPE:", mean_absolute_percentage_error(true_missing, predicted_missing))
print("R2:", r2_score(true_missing, predicted_missing))


# # ZIFA MODEL (commented out for faster REMAP testing)
# zifa_recon, X_subset, mask_subset = zifa_impute(tensor, mask)

# zifa_pred = zifa_recon[mask_subset == 0]
# zifa_true = X_subset[mask_subset == 0]

# valid = ~np.isnan(zifa_pred)

# zifa_pred = zifa_pred[valid]
# zifa_true = zifa_true[valid]

# print("Num eval points:", len(zifa_true))

# if len(zifa_true) == 0:
#     print("\n--- ZIFA Results ---")
#     print("No valid values to evaluate after filtering.")
# else:
#     print("\n--- ZIFA Results ---")
#     print("MAE:", mean_absolute_error(zifa_true, zifa_pred))
#     print("MAPE:", mean_absolute_percentage_error(zifa_true, zifa_pred))
#     print("R2:", r2_score(zifa_true, zifa_pred))

# REMAP MODEL
print("\n--- Testing REMAP ---")
try:
    # Set up REMAP parameters
    remap_args = build_remap_args(low_rank=10, max_iter=20, weight=0.1, imp=0.1, reg=0.1)
    
    # Run REMAP imputation
    remap_recon, remap_U, remap_V = remap_impute_tensor(tensor, mask, chem_sim=None, prot_sim=None, args=remap_args)
    
    print(f"REMAP reconstruction shape: {remap_recon.shape}")
    
    # Reshape back to original tensor shape and extract masked values
    remap_recon_tensor = remap_recon.reshape(tensor.shape)
    
    remap_pred = remap_recon_tensor[mask == 0]
    remap_true = tensor[mask == 0]
    
    # Only evaluate meaningful values
    valid_remap = remap_true != 0
    remap_pred = remap_pred[valid_remap]
    remap_true = remap_true[valid_remap]
    
    if len(remap_true) == 0:
        print("\n--- REMAP Results ---")
        print("No valid values to evaluate.")
    else:
        print("\n--- REMAP Results ---")
        print("MAE:", mean_absolute_error(remap_true, remap_pred))
        print("MAPE:", mean_absolute_percentage_error(remap_true, remap_pred))
        print("R2:", r2_score(remap_true, remap_pred))
except Exception as e:
    print(f"REMAP failed with error: {e}")