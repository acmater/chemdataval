import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import pickle
import scipy.stats as st
from .preprocessing import *

default_kernel = RBF(1) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-20, 100000.0))

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

def normalise(arr):
    """
    Simple helper function to normalise and return an array.
    """
    return arr/np.linalg.norm(arr)

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

def test_model(model, X, Y, train_idx, test_idx, *args, **kwargs):
    """
    Generic model testing function


    Parameters
    ----------
    model : sklearn.MixinRegressor
    
    """
    check_invalid_selection(train_idx, test_idx)

    X, Y = preprocess(X, Y, train_idx)
    model = model(*args, **kwargs)
    model.fit(X[train_idx], Y[train_idx])
    return model.score(X[test_idx], Y[test_idx])
           
def make_cov(variance_vector):
    cov = np.zeros((len(variance_vector), len(variance_vector)))
    np.fill_diagonal(cov, variance_vector)
    return cov   

def assess_gpr(X, Y, train_idx, test_idx, size=50, kernel=default_kernel, runs=100, inform=None, random=False, T=1000):
    scores = []
    for i in tqdm.tqdm(range(runs)):
        if random:
            train_idxs = np.random.choice(train_idx, size=(size,),replace=False)
        else:
            averages = np.mean(rep_informativeness(kf_results,strategy="Random", seed_size=50, runs=100, subset_idxs=None, dset_size=239), axis=0) * T
            train_idxs = np.random.choice(train_idx, size=(size,),replace=False, p=(np.exp(averages) / np.sum(np.exp(averages))))
        assert len(np.intersect1d(train_idxs, test_idx)) == 0, "There is overlap with the test set" 
        scores.append(test_gpr(X, Y, train_idxs, test_idx, kernel=kernel))
    return scores

def gaussian_kde_plot(scores, vals = np.linspace(0,1,1000), ax=None, label=""):
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
    Simpler helper function to compare two sets of indices and ensure that there is no overlap between them.
    
    Parameters
    ----------
    chosen : [int]
        The indices that have been selected.
        
    holdout : [int]
        The indices that are not allowed to be selected. 
    """
    assert len(np.intersect1d(chosen, holdout)) == 0, "The method is selecting held out data points."
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
    

def test():
	print("Hello")


	a = np.array([5])
	print(a.square())
