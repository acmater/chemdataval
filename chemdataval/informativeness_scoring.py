import numpy as np
from .utils import ensure_array


def masked_values(func):
    """
    Masking decorator that replaces all zero values with a mask from the output
    of any of the informativeness scoring methods.
    """

    def inner(*args, **kwargs):
        inform = np.ma.masked_equal(func(*args, **kwargs), 0)
        return inform

    return inner


@masked_values
def informativeness_scoring(
    results, dset_size, seed_size=30, subset_idxs=None, scaler=None,
):
    """
    Computes the informativeness of each point within an active run.

    Parameters
    ----------
    results : {'idxs' : np.array,
               'scores' : np.array}
        Results for an individual representation, fold, and sampling strategy.

    dset_size : int, default=266
        The overall size of the dataset

    subset_idxs : np.array[int], default=None
        The indices of the subset that the provided indices correspond to.

    seed_size : int, default=30
        The seed size that was selected, used to remove seed indices from the
        indice arrays

    Returns
    -------
    informativeness : np.ma.array(dset_size, results)
    """
    scores, idxs = ensure_array(results["scores"]), ensure_array(results["idxs"])
    scores, idxs = np.atleast_2d(scores), np.atleast_2d(idxs)

    # Re-indexes the indices so that they match the provided subset.
    if subset_idxs is not None:
        idxs = subset_idxs[idxs]

    diffs = np.array([np.diff(score) for score in scores])
    informativeness = np.zeros((diffs.shape[0], dset_size))

    for run in range(diffs.shape[0]):
        scaled_diff = diffs[run]
        if scaler is not None:
            scaled_diff = scaler(scaled_diff)
        informativeness[run, idxs[run][seed_size:]] = scaled_diff

    return informativeness


@masked_values
def fold_informativeness(
    rep_results,
    strategy,
    runs,
    dset_size,
    folds=5,
    scaler=None,
    seed_size=30,
    subset_idxs=None,
):
    """
    Wrapper function that computes the informativeness over all folds through multiple calls to informativeness_scoring. All parameters are the same, except the following

    rep_results {"Fold n" : {strategy : results}}
        The results to be passed

    strategy : str, default="Active"
        The sampling strategy that used used. Used to index the rep_results dictionary provided
    """
    informativeness = np.zeros((folds * runs, dset_size))

    for idx in range(folds):
        fold_values = informativeness_scoring(
            rep_results[f"Fold {idx+1}"][strategy],
            seed_size=seed_size,
            scaler=scaler,
            dset_size=dset_size,
            subset_idxs=subset_idxs,
        )
        informativeness[idx * runs : idx * runs + runs] = fold_values

    return informativeness


@masked_values
def rep_informativeness(
    kf_results,
    strategy,
    runs,
    dset_size,
    folds=5,
    scaler=None,
    seed_size=30,
    subset_idxs=None,
):
    """
    Wrapper function that computes the informativeness over all representations of the data through repeated calls to fold informativeness.
    Function arguments are the same.
    """
    informativeness = np.zeros((len(kf_results) * folds * runs, dset_size))
    for idx, rep in enumerate(kf_results.keys()):
        informativeness[
            idx * (folds * runs) : idx * (folds * runs) + (folds * runs)
        ] = fold_informativeness(
            kf_results[rep],
            strategy=strategy,
            seed_size=seed_size,
            runs=runs,
            folds=folds,
            scaler=scaler,
            dset_size=dset_size,
            subset_idxs=subset_idxs,
        )
    return informativeness


@masked_values
def strategy_informativeness(
    kf_results,
    strategies,
    rep,
    runs,
    dset_size,
    folds=5,
    scaler=None,
    seed_size=30,
    subset_idxs=None,
):
    """
    Wrapper function that computes the informativeness over all sampling strategies of the data for a given representation.
    """
    informativeness = np.zeros((len(strategies) * folds * runs, dset_size))
    for idx, strategy in enumerate(strategies.keys()):
        informativeness[
            idx * (folds * runs) : idx * (folds * runs) + (folds * runs)
        ] = fold_informativeness(
            kf_results[rep],
            strategy=strategy,
            seed_size=seed_size,
            runs=runs,
            folds=folds,
            scaler=scaler,
            dset_size=dset_size,
            subset_idxs=subset_idxs,
        )
    return informativeness


@masked_values
def complete_informativeness(
    kf_results,
    strategies,
    folds=5,
    runs=5,
    dset_size=266,
    scaler=None,
    seed_size=30,
    subset_idxs=None,
):
    informativeness = np.zeros(
        (len(kf_results) * len(strategies) * folds * runs, dset_size)
    )
    for idx, strat in enumerate(strategies.keys()):
        informativeness[
            idx
            * (folds * runs * len(kf_results)) : idx
            * (folds * runs * len(kf_results))
            + folds * runs * len(kf_results)
        ] = rep_informativeness(
            kf_results,
            strategy=strat,
            seed_size=seed_size,
            runs=runs,
            folds=folds,
            scaler=scaler,
            dset_size=dset_size,
            subset_idxs=subset_idxs,
        )
    return informativeness


def correlation_matrix(
    results, strategy="Random", seed_size=50, runs=100, subset_idxs=None
):
    informativeness_array = np.array(
        [
            np.mean(
                fold_informativeness(
                    results[rep],
                    strategy=strategy,
                    seed_size=50,
                    runs=runs,
                    subset_idxs=subset_idxs,
                ),
                axis=0,
            )
            for rep in results.keys()
        ]
    )
    return np.corrcoef(informativeness_array)


def correlation_matrix_rep(
    results, strategies, representation="CM", seed_size=50, runs=100, subset_idxs=None
):
    informativeness_array = np.array(
        [
            np.mean(
                fold_informativeness(
                    results[representation],
                    strategy=strat,
                    seed_size=50,
                    runs=runs,
                    subset_idxs=subset_idxs(),
                ),
                axis=0,
            )
            for strat in strategies.keys()
        ]
    )
    return np.corrcoef(informativeness_array)
