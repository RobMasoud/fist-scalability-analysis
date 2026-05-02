import numpy as np

def zifa_impute(tensor, mask, sparsity_thresh=0.95, max_genes=None, latent_dim=10):
    """
    Imputes missing values using ZIFA (Zero-Inflated Factor Analysis).
    
    Args:
        tensor: 3D gene expression tensor (samples x spatial_dim1 x genes)
        mask: binary mask (1=observed, 0=missing) with same shape as tensor
        sparsity_thresh: filter out genes with >X% zeros (default 0.95 = 95%)
        max_genes: max number of genes to use (None = use all after filtering)
        latent_dim: number of latent dimensions for ZIFA (default 10)
    
    Returns:
        reconstructed: imputed gene expression (log scale), shape (samples, filtered_genes)
        X_filtered: original filtered input (log scale), shape (samples, filtered_genes)
        mask_filtered: mask for filtered genes, shape (samples, filtered_genes)
    """
    import models.zifa.ZIFA.ZIFA.block_ZIFA as block_ZIFA

    original_shape = tensor.shape

    # Reshape 3D tensor to 2D (samples × genes) for ZIFA input
    X = tensor.reshape(-1, original_shape[-1]).astype(float)
    mask_2d = mask.reshape(-1, original_shape[-1])

    print(f"Original tensor shape: {X.shape}")

    # Apply mask: set missing values to 0
    X_masked = X.copy()
    X_masked[mask_2d == 0] = 0

    # Log transform for zero-inflated data
    X_masked = np.log1p(X_masked)

    # Filter genes with too many zeros to improve performance
    zero_fraction = np.mean(X_masked == 0, axis=0)
    keep_cols = zero_fraction < sparsity_thresh
    
    X_filtered = X_masked[:, keep_cols]
    mask_filtered = mask_2d[:, keep_cols]

    print(f"After filtering genes with >{sparsity_thresh*100:.0f}% zeros: {X_filtered.shape}")

    # Optionally cap number of genes for memory constraints
    if max_genes is not None and X_filtered.shape[1] > max_genes:
        print(f"Capping genes from {X_filtered.shape[1]} to {max_genes}")
        X_filtered = X_filtered[:, :max_genes]
        mask_filtered = mask_filtered[:, :max_genes]

    print(f"ZIFA input shape: {X_filtered.shape}")

    # Run ZIFA EM algorithm
    Z, params = block_ZIFA.fitModel(X_filtered, latent_dim)

    # Extract reconstructed data from ZIFA params
    reconstructed = np.expm1(params['X'])
    
    print(f"Reconstructed shape: {reconstructed.shape}")

    return reconstructed, np.expm1(X_filtered), mask_filtered