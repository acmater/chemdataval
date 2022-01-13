import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess(X, Y, train_index, n_components=50):
    """
    Performs complete pre-processing of pca dimensionality reduction followed by standardisation.
    """
    X = pca_process(X, train_index, n_components)
    X̂, Ŷ = standardise(X, Y, train_index)
    return X̂, Ŷ
    
def pca_process(X, train_index, n_components=50):
    """
    Performs PCA dimensionality reduction using only the training data.
    """
    pca_ = PCA(n_components=n_components).fit(X[train_index])
    return pca_.transform(X)
    
def standardise(X, Y, train_index, return_scaler=False):
    """
    Performs simple pre-processing using the standard scaler and fitting the the training data alone.
    
    X : np.ndarray(N x M)
        data representations (N samples with M features)
    Y : np.ndarray(N,)
        data values
    train_idx : np.ndarray(D, dtype=np.int)
        The training indices
    """
    datascaler = StandardScaler().fit(X[train_index])
    predsscaler = StandardScaler().fit(Y[train_index].reshape(-1,1))

    X̂ = datascaler.transform(X) 
    Ŷ = predsscaler.transform(Y.reshape(-1,1))
    
    if return_scaler:
        return X̂, Ŷ, datascaler, predsscaler
    else:
        return X̂, Ŷ

