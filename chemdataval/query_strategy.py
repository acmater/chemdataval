import numpy as np
from .utils import kmax, kmin, normalise

def modify_std(std, density_vector, β, train_index, pool_idxs, normalise):
    """
    Helper function to modify the standard deviation in accordance with the standard deviation. Separated out of other codebase to provide independenting testing
    of this method. Uses precisely the same function signature as GP_regression_std.
    """
    if normalise:
        std /= np.linalg.norm(std)
       
    if density_vector is not None:
        if normalise:
            density_vector /= np.linalg.norm(density_vector)
        # This line sucks. Basically the pool indices are a subset of the train indices, and you only want the relevant values for the density vector.
        # The problem is that the density vector has been formed from train_index, but then had its indices reset.
        # So, you need to find the indices within train index that are also in pool index, and then grab those out and use them to index the density vector
        # Simple...
        std = std * np.power(density_vector[np.in1d(train_index,pool_idxs).nonzero()], β)

    return std

def GP_regression_std(regressor, X, pool_idxs, train_index=None, density_vector=None, β=1.0, normalise=False, batch_size=1):
    """
    The goal of this function is to cache the computation of the densities so that methods such as the REMatch kernel can be used.
    
    Parameters
    ----------
    X : np.array (N,M)
        The full representations (M dimensions) of the entire training dataset (N samples).
        
    regressor : model
        Machine learning model that must have a valid predict method that mirrors the functionality observed in scikit-learn.
    
    pool_idxs : np.array(K,)
        The indices of the samples that the model is allows to sample.
    
    train_index : np.array(N,), default=None
        Training index that must be provided with the density vector to ensure the dimensions are correct.
    
    density_vector : np.array(N,), default=None
        The density vector that is used to scale the uncertainty terms.
        
    β : float, default=1.0
        A scaling parameter that controls how important the density term is.
        
    normalise : bool, default=False
        Whether or not to normalise the vectors prior to multiplying them.
        
    batch_size : int, default=1
        The batch size of samples to be greedily selected.
    """
    if density_vector is not None:
        assert len(X[train_index]) == len(density_vector), "The density vector must have the same dimensions as X"
        
    assert len(pool_idxs) <= len(X), "Pool indexes must be less than or equal to X."
    
    _, std = regressor.predict(X[pool_idxs], return_std=True)
    
    std = modify_std(std=std,
                     density_vector=density_vector,
                     β=β,
                     train_index=train_index,
                     pool_idxs=pool_idxs,
                     normalise=normalise)
        
    if batch_size == 1:
        sub_idx = np.argmax(std)
    else:
        sub_idx = kmax(std, batch_size, indices=True)
        
    # Re-index the sub-index using the provided pool indices, so that the selected index correctly matches those of the original representation array, X.
    query_idx = pool_idxs[sub_idx]
    return query_idx, X[query_idx], sub_idx
   
