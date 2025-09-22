"""Helper functions"""
from itertools import chain
import pandas as pd
import numpy as np
import scipy.stats as ss
import polars as pl


def find_feat_cols(lframe):
    return [col for col in lframe.columns if not col.startswith('Metadata_')]

def find_meta_cols(lframe):
    return [col for col in lframe.columns if col.startswith('Metadata_')]

def remove_nan_infs_columns(dframe: pd.DataFrame):
    """Remove columns with NaN and INF"""
    feat_cols = find_feat_cols(dframe)
    withnan = dframe[feat_cols].isna().sum()[lambda x: x > 0]
    withinf = (dframe[feat_cols] == np.inf).sum()[lambda x: x > 0]
    withninf = (dframe[feat_cols] == -np.inf).sum()[lambda x: x > 0]
    redlist = set(chain(withinf.index, withnan.index, withninf.index))
    dframe_filtered = dframe[[c for c in dframe.columns if c not in redlist]]
    return dframe_filtered

def rank_int_array(array: np.ndarray,
                   c: float = 3.0 / 8,
                   stochastic: bool = True,
                   seed: int = 0):
    '''
    Perform rank-based inverse normal transformation in a 1d numpy array. If
    stochastic is True ties are given rank randomly, otherwise ties will share
    the same value.

    Copied directly from: https://github.com/carpenter-singh-lab/2023_Arevalo_NatComm_BatchCorrection/blob/cd9bcf99240880a5c9f9858debf70e94f5b4c0f7/preprocessing/transform.py#L8
    Adapted from: https://github.com/edm1/rank-based-INT/blob/85cb37bb8e0d9e71bb9e8f801fd7369995b8aee7/rank_based_inverse_normal_transformation.py
    '''
    rng = np.random.default_rng(seed=seed)

    if stochastic:
        # Shuffle
        ix = rng.permutation(len(array))
        rev_ix = np.argsort(ix)
        array = array[ix]
        # Get rank, ties are determined by their position(hence why we shuffle)
        rank = ss.rankdata(array, method="ordinal")
        rank = rank[rev_ix]
    else:
        # Get rank, ties are averaged
        rank = ss.rankdata(array, method="average")

    x = (rank - c) / (len(rank) - 2 * c + 1)
    return ss.norm.ppf(x)

def inverse_normal_transform(data, feat_cols=None, c: float = 3.0 / 8, stochastic: bool = True, seed: int = 0):
    """
    Apply inverse normal transformation (rank-based normalization) to data using rank_int_array.
    
    Parameters:
    -----------
    data : polars.DataFrame
       Input dataframe
    feat_cols : list, optional
       Specific columns to transform. If None, auto-detect numeric columns
    c : float
       Blom's constant for rank transformation
    stochastic : bool
       Whether to use stochastic tie-breaking
    seed : int
       Random seed for stochastic tie-breaking
    """
    from tqdm.contrib.concurrent import thread_map
    import scipy.stats as ss
    
    # Determine columns to transform
    if feat_cols is not None:
        numeric_cols = feat_cols
    else:
        numeric_cols = [col for col in data.columns if "Meta" not in col]
    
    if not numeric_cols:
        return data  # No columns to transform
    
    # Convert to numpy for processing
    data_np = data.select(numeric_cols).to_numpy()
    
    def to_normal(i):
        """Transform column i using rank_int_array"""
        col_data = data_np[:, i].copy()  # Work on copy to avoid race conditions
        mask = ~np.isnan(col_data)
        
        if np.sum(mask) > 0:  # Only transform if there are non-null values
            # Transform non-null values
            transformed_values = rank_int_array(
               col_data[mask], 
               c=c, 
               stochastic=stochastic, 
               seed=seed + i  # Different seed per column for stochastic mode
            ).astype(np.float32)
            
            # Put transformed values back
            col_data[mask] = transformed_values
            data_np[:, i] = col_data
    
    # Apply transformation to all columns in parallel
    thread_map(to_normal, range(len(numeric_cols)), leave=False)
    
    # Convert back to polars DataFrame
    transformed_df = pl.DataFrame(
       data_np, 
       schema={col: pl.Float32 for col in numeric_cols}
    )
    
    # Combine with non-numeric columns if they exist
    non_numeric_cols = [col for col in data.columns if col not in numeric_cols]
    
    if non_numeric_cols:
        # Preserve original column order
        result_df = data.select(non_numeric_cols).hstack(transformed_df)
        return result_df.select(data.columns)  # Maintain original column order
    else:
        return transformed_df