import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def preprocess(X, Y, train_index, n_components=50):
    """
    Performs complete pre-processing of pca dimensionality reduction followed by standardisation.
    """
    X̂, Ŷ = standardise(X, Y, train_index)
    X̂ = pca_process(X̂, train_index, n_components)
    return X̂, Ŷ


def pca_process(X, train_index, n_components=50):
    """
    Performs PCA dimensionality reduction using only the training data.
    """
    pca_ = PCA(n_components=n_components).fit(X[train_index])
    return pca_.transform(X)


def standardise(X, Y=None, train_index=None, return_scaler=False):
    """
    Performs simple pre-processing using the standard scaler and fitting the the training data alone.

    X : np.ndarray(N x M)
        data representations (N samples with M features)
    Y : np.ndarray(N,)
        data values
    train_idx : np.ndarray(D, dtype=np.int)
        The training indices
    """
    assert len(X) > 0, "X cannot be empty."

    if train_index is None:
        train_index = range(len(X))

    datascaler = StandardScaler().fit(X[train_index])
    X̂ = datascaler.transform(X)

    if Y is not None:
        predsscaler = StandardScaler().fit(Y[train_index].reshape(-1, 1))
        Ŷ = predsscaler.transform(Y.reshape(-1, 1))

        if return_scaler:
            return X̂, Ŷ, datascaler, predsscaler
        else:
            return X̂, Ŷ

    else:
        if return_scaler:
            return X̂, datascaler
        else:
            return X̂
