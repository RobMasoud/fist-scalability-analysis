import sys
from pathlib import Path
import argparse
import numpy as np

# Add REMAP_py to sys.path so we can import the REMAP module
remap_path = Path(__file__).resolve().parent / "remap" / "REMAP_py"
if str(remap_path) not in sys.path:
    sys.path.insert(0, str(remap_path))
    
from models.remap.REMAP_py.REMAP import REMAP as run_remap

def build_remap_args(
    low_rank=10,
    max_iter=20,
    weight=0.1,
    imp=0.1,
    reg=0.1,
    weight_chem=0.75,
    weight_prot=0.25,
    seed=1987,
):
    """
    Build a simple argparse.Namespace for REMAP parameters.
    
    Args:
        low_rank: number of latent factors (default 10)
        max_iter: maximum iterations for optimization (default 20)
        weight: global weight parameter (default 0.1)
        imp: global imputation parameter (default 0.1)
        reg: regularization parameter (default 0.1)
        weight_chem: importance weight for chem-chem similarity (default 0.75)
        weight_prot: importance weight for prot-prot similarity (default 0.25)
        seed: random seed (default 1987)
    
    Returns:
        argparse.Namespace: object with REMAP parameters
    """
    return argparse.Namespace(
        low_rank=low_rank,
        max_iter=max_iter,
        weight=weight,
        imp=imp,
        reg=reg,
        weight_chem=weight_chem,
        weight_prot=weight_prot,
        seed=seed,
    )

def remap_impute(R, chem_sim=None, prot_sim=None, args=None):
    """
    Run REMAP on a 2D input matrix.
    
    Args:
        R: 2D matrix (samples x genes or samples x features)
        chem_sim: similarity matrix for the first dimension, or None
        prot_sim: similarity matrix for the second dimension, or None
        args: argparse.Namespace with REMAP parameters
    
    Returns:
        reconstructed: U @ V.T prediction matrix
        U: low-rank sample factors
        V: low-rank feature factors
    """
    if args is None:
        args = build_remap_args()

    if not isinstance(R, np.ndarray):
        R = np.array(R, dtype=np.float64)

    U, V = run_remap(R, chem_sim, prot_sim, args)
    reconstructed = np.array(U @ V.T, dtype=np.float64)
    return reconstructed, U, V

def tensor_to_matrix(tensor):
    """
    Convert a 3D tensor to a 2D matrix for REMAP/ZIFA-style input.
    
    Args:
        tensor: 3D tensor (e.g., spatial x spatial x genes)
    
    Returns:
        2D matrix: flattened tensor (samples x genes)
    """
    original_shape = tensor.shape
    return tensor.reshape(-1, original_shape[-1]).astype(float)

def remap_impute_tensor(tensor, mask, chem_sim=None, prot_sim=None, args=None):
    """
    Run REMAP on a 3D tensor with a binary mask.
    
    Args:
        tensor: 3D gene expression tensor
        mask: binary mask (1=observed, 0=missing)
        chem_sim: similarity matrix for samples, or None
        prot_sim: similarity matrix for genes, or None
        args: argparse.Namespace with REMAP parameters
    
    Returns:
        reconstructed: imputed matrix (samples x genes)
        U: low-rank sample factors
        V: low-rank gene factors
    """
    X = tensor_to_matrix(tensor)
    mask_2d = mask.reshape(X.shape)
    X_masked = X.copy()
    X_masked[mask_2d == 0] = 0.0  # Apply mask by setting missing to 0
    return remap_impute(X_masked, chem_sim=chem_sim, prot_sim=prot_sim, args=args)
