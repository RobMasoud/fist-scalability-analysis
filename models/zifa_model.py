import numpy as np

def zifa_impute(tensor, mask, max_samples=2000):
    import models.zifa.ZIFA.ZIFA.block_ZIFA as block_ZIFA

    original_shape = tensor.shape

    X = tensor.reshape(-1, original_shape[-1]).astype(float)
    mask_2d = mask.reshape(-1, original_shape[-1])

    X = X[:max_samples]
    mask_2d = mask_2d[:max_samples]

    X_masked = X.copy()
    X_masked[mask_2d == 0] = 0

    X_masked = np.log1p(X_masked)

    zero_fraction = np.mean(X_masked == 0, axis=0)
    keep_cols = zero_fraction < 0.90

    X_filtered = X_masked[:, keep_cols]
    mask_filtered = mask_2d[:, keep_cols]

    if X_filtered.shape[1] > 6000:
        X_filtered = X_filtered[:, :6000]
        mask_filtered = mask_filtered[:, :6000]

    Z, params = block_ZIFA.fitModel(X_filtered, 10)

    reconstructed = np.expm1(params['X'])
    
    print("ZIFA input shape:", X_filtered.shape)
    print("Reconstructed shape:", reconstructed.shape)

    return reconstructed, np.expm1(X_filtered), mask_filtered