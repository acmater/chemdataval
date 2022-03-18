from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from modAL.density import information_density
from scipy.spatial.distance import cdist
import tqdm

from .query_strategy import *
from .utils import *
from .preprocessing import *

import warnings

kernel = RBF(1) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-20, 100000.0))

def active_test(seed, X, Y, train_index, test, queries=180, kernel=RBF(1), query=GP_regression_std, density_vector=None, β=1):
    """
    Code to test the active learning system when provided with a seed array, a regressor, and a query strategy.
    """
    # Compute the pool which stores all samples not in the seed.
    pool_idxs = np.setdiff1d(train_index,seed)
    
    if queries == None:
        queries = len(pool_idxs)

    # Initialise the regressor, with the desired kernel.
    regressor = ActiveLearner(
        estimator=GaussianProcessRegressor(kernel=kernel),
        query_strategy=query,
        X_training=X[seed], y_training=Y[seed].reshape(-1,1))
    
    # You want the scores to contain a baseline score value
    idxs, scores = list(seed), [regressor.score(*test)]
    for _ in range(queries):
        query_idx, query_instance, sub_idx = regressor.query(X, pool_idxs, train_index, density_vector, β=β)
        
        idxs.append(query_idx)
        regressor.teach(X[query_idx].reshape(1,-1), Y[query_idx].reshape(1,-1))    
        scores.append(regressor.score(*test))
        
        # Delete the row of the pool using the sub_idx as otherwise they can be selected again.
        pool_idxs = np.delete(pool_idxs, sub_idx, axis=0)
  
    return idxs, scores

def random_test(seed, X, Y, train_index, test, queries=180, kernel=RBF(1),density_vector=None, β=None):
    pool = list(np.setdiff1d(list(train_index), seed))
    seed = list(seed)
    
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X[seed], Y[seed])
    scores = [gpr.score(*test)]
    
    for idx in range(queries):
        # Choose the index of the item in pool
        choice = int(np.random.choice(range(len(pool)), size=(1,)))
        seed.append(pool[choice])
        
        # Delete the item from pool
        del pool[choice]
                     
        gpr = GaussianProcessRegressor(kernel=kernel)
        gpr.fit(X[seed], Y[seed])
    
        scores.append(gpr.score(*test))
    
    return seed, scores

def fold_testing(X, Y, strategies, kf, categorical=None,random_runs=2, seed_size=30, queries=None, random_state_seed=10, debug=True, kernel=kernel, n_components=50, β=1, subset_idxs=None):
    """
    Conducts full fold testing for a given strategy across the dataset. 
    
    Parameters 
    ----------
    
    subset_idxs : np.ndarray
        The integer indices of the subset of the data
    """
    kf_results = {}
    
    with warnings.catch_warnings():
        # Remove the convergence and nearing bound warnings that fill up the output section
        warnings.simplefilter("ignore")
        
        for kf_idx,item in enumerate(kf.split(X,categorical)):

            print(f"Running fold {kf_idx+1}")
            train_index, test_index = item
            
            # Standardise the data using only the train components.
            X̂, Ŷ = preprocess(X, Y, train_index, n_components=n_components)
            
            test = (X̂[test_index], Ŷ[test_index])
            if queries is None:
                queries = len(train_index) - seed_size

            results = {}
            results["Test indexes"] = test_index
            for strategy in strategies.keys():
                results[strategy] = {key : arr for key,arr in zip(["idxs", "scores"],
                                                                  [np.zeros((random_runs,queries+seed_size),dtype=np.int),np.zeros((random_runs,queries+1))])}
                                                                  # The +1 is to factor in the first run
            if debug:
                print(f"Training set size: {len(train_index)}")
                print(f"Testing set size: {len(test_index)}")
                print("Strategies:")
                for strategy in strategies:
                    print(f"\t{strategy}")
                print(f"Queries: {queries}")
                       
            density_arrays = {"Information Density" : 1/(1+cdist(X̂,X̂,metric="euclidean")),
                              "Information Density 2.0" : 1/(1+cdist(X̂,X̂,metric="euclidean"))}            

            # Recompute the density vectors so that they only utilise the training information in the computation.
            # density_array[train_index,:][:,train_index] selects only the rows and columns that are in the train index
            density_vectors = {name : np.mean(density_array[train_index,:][:,train_index],axis=1) for name, density_array in density_arrays.items()}
   
            for idx in tqdm.tqdm(range(random_runs)):
                # Choose the random seed that will be shared by density, active, and random.
                seed = np.random.choice(train_index, size=(seed_size,), replace=False)
                               
                for strategy, strategy_func in strategies.items():                      
                    if strategy == "KMean":
                        seed = kmean_seed(X̂, seed_size, train_index, random_state=idx)
                    
                    density_vector = density_vectors.get(strategy, None)
                    
                    idxs, scores = strategy_func(seed, X̂, Ŷ, train_index, test, queries=queries, kernel=kernel, density_vector=density_vector)
                    
                    # Sanity check tests
                    assert len(np.intersect1d(idxs, test_index)) == 0, f"Fold {kf_idx+1} - {strategy} There is overlap between the training and testing indices."
                    assert len(idxs) == len(np.unique(idxs)), f"Fold {kf_idx+1} - {strategy} There are duplicates in the training indices selected."
                    assert len(np.intersect1d(idxs[:seed_size], idxs[seed_size:])) == 0, f"Fold {kf_idx+1} - {strategy} The model is re-selecting points in the seed dataset."

                    results[strategy]["idxs"][idx,:] = np.array(idxs)
                    results[strategy]["scores"][idx, :] = np.array(scores)

            kf_results[f"Fold {kf_idx + 1}"] = results
            
        return kf_results
