import numpy as np

def zifa_impute(tensor, mask, sparsity_thresh=0.95, max_genes=None, latent_dim=10):
    import models.zifa.ZIFA.ZIFA.block_ZIFA as block_ZIFA

    original_shape = tensor.shape

    X = tensor.reshape(-1, original_shape[-1]).astype(float)
    mask_2d = mask.reshape(-1, original_shape[-1])

    print(f"Original tensor shape: {X.shape}")

    X_masked = X.copy()
    X_masked[mask_2d == 0] = 0

    X_masked = np.log1p(X_masked)

    zero_fraction = np.mean(X_masked == 0, axis=0)
    keep_cols = zero_fraction < sparsity_thresh
    
    X_filtered = X_masked[:, keep_cols]
    mask_filtered = mask_2d[:, keep_cols]

    print(f"After filtering genes with >{sparsity_thresh*100:.0f}% zeros: {X_filtered.shape}")

    if max_genes is not None and X_filtered.shape[1] > max_genes:
        print(f"Capping genes from {X_filtered.shape[1]} to {max_genes}")
        X_filtered = X_filtered[:, :max_genes]
        mask_filtered = mask_filtered[:, :max_genes]

    print(f"ZIFA input shape: {X_filtered.shape}")

    Z, params = block_ZIFA.fitModel(X_filtered, latent_dim)

    reconstructed = np.expm1(params['X'])
    
    print(f"Reconstructed shape: {reconstructed.shape}")

    return reconstructed, np.expm1(X_filtered), mask_filtered