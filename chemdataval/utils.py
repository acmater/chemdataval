import numpy as np

def data_stats(data, axis=0):
    """
    Computes the mean and standard deviation of the data provided and returns it.
    """
    μ = np.mean(data, axis=axis)
    σ = np.sqrt(np.var(data, axis=axis))
    return μ, σ

def kmax(arr, k, indices=False, sort=True):
    """
    Numpy helper function which returns to k maximum values of a list
    """
    idxs = np.argpartition(arr, -k)[-k:]
    return kindices(idxs, arr, indices, sort)
    
def kmin(arr, k, indices=False, sort=True):
    """
    Numpy helper function which returns to k minimum values of a list
    """
    idxs = np.argpartition(arr, k)[:k]
    return kindices(idxs, arr, indices, sort)
    
def kindices(idxs, arr, indices=False, sort=True):
    """
    Helper function that manages control flow branching for kmin and kmax.
    """
    if indices:
        return idxs
    if sort:
        return np.sort(arr[idxs])
    else:
        return arr[idxs]
