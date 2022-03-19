import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
import pickle
import scipy.stats as st

default_kernel = RBF(1) + WhiteKernel(
    noise_level=0.1, noise_level_bounds=(1e-20, 100000.0)
)


def data_stats(data, axis=0):
    """
    Computes the mean and standard deviation of the data provided and returns
    it.
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


def normalise(arr):
    """
    Simple helper function to normalise and return an array.
    """
    return arr / np.linalg.norm(arr)


def save_obj(obj, name):
    assert isinstance(name, str), "Name must be a string"
    ext = name.split(".")[-1]
    if ext != "pkl":
        ext = ".pkl"
    with open(name + ext, "wb") as f:
        pickle.dump(obj, f)
    print("File Saved Successfully")
    return None


def load_obj(name):
    assert isinstance(name, str), "Name must be a string"
    ext = name.split(".")[-1]
    if ext == "pkl":
        ext = ""
    else:
        ext = ".pkl"
    with open(name + ext, "rb") as f:
        obj = pickle.load(f)
    print("File Loaded Successfully")
    return obj


def make_cov(variance_vector):
    cov = np.zeros((len(variance_vector), len(variance_vector)))
    np.fill_diagonal(cov, variance_vector)
    return cov


def gaussian_kde_plot(scores, vals=np.linspace(0, 1, 1000), ax=None, label=""):
    kernel = st.gaussian_kde(scores)
    ax.plot(vals, kernel(vals), label=label)
    ax.fill_between(vals, 0, kernel(vals), alpha=0.2)


def softmax(arr):
    """
    Computes the softmax.
    """
    return np.exp(arr) / np.sum(np.exp(arr))


def prob_vec(arr, T=1):
    """
    Scales an array by a temperature value before computing the softmax.
    """
    arr *= T
    return softmax(arr)


def check_invalid_selection(chosen, holdout):
    """
    Simpler helper function to compare two sets of indices and ensure that
    there is no overlap between them.

    Parameters
    ----------
    chosen : [int]
        The indices that have been selected.

    holdout : [int]
        The indices that are not allowed to be selected.
    """
    assert (
        len(np.intersect1d(chosen, holdout)) == 0
    ), "The method is selecting held out data points."
    return None


def ensure_array(arr):
    """
    Helper function that ensures that an iterable is indeed an array
    """
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.array(arr)
        except Exception as e:
            print(e)
    return arr


def kmean_seed(X, seed_size, train_index, random_state=0):
    """
    X should only be the training data
    """
    kmean_centroid_in_train = (
        KMeans(n_clusters=seed_size, random_state=random_state)
        .fit(X[train_index])
        .cluster_centers_
    )
    # Only allow it to select the nearest points from the training data.
    sub_query = vq(kmean_centroid_in_train, X[train_index])[0]
    kmean_seed_ = train_index[sub_query]
    i = 1

    # This loop is to ensure that the kmean seed does not
    # contain any duplicates
    while len(np.unique(kmean_seed_)) < seed_size:
        kmean_centroid_in_train = (
            KMeans(n_clusters=seed_size + i, random_state=i).fit(X).cluster_centers_
        )
        sub_query = vq(kmean_centroid_in_train, X[train_index])[0]
        kmean_seed_ = train_index[sub_query]
        i += 1

    return np.unique(kmean_seed_)[:seed_size]
