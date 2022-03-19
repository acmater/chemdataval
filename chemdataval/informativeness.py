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
def index_scores(idxs, scores, seed_size, dset_size):
    """
    Takes an array of indices, scores, and a seed size, and aligns the changes
    or differences in scores to the correct indices

    Parameters
    ----------
    idxs : np.array(N + seed_size, )
        The indices of each array. Note that up to seed_size, these are ignored.
        It is from the start of seed size that they correlate to scores.

    scores : np.array(N)
        The scores associated with training a model. The i^th value of scores
        corresponds to a model trained on the first seed_size+(i-1)^th values
        idxs.

    seed_size : int
        The seed_size that the first model was trained on. This is important as
        the seeds are included in the idxs, so this provides the offset to grab
        the correct values.

    dset_size : int
        An integer for dataset size. This is used to initalise a blank numpy
        array that is filled with the appropriate difference values.

    Returns
    -------
    MC_shapleys : np.array(dset_size, )
        The Monte Carlo Shapley estimates for this particular run/permutation
    """
    # The -1 below is because the scores include training on just the seed set.
    assert len(scores) - 1 + seed_size == len(
        idxs
    ), "There is a mismatch between the dimensions of scores, indices, and seed size."

    differences = np.diff(scores)
    MC_shapleys = np.zeros(dset_size,)
    trained_idxs = idxs[seed_size:]
    MC_shapleys[trained_idxs] = differences
    return MC_shapleys


@masked_values
def informativeness_scoring(
    idxs, scores, dset_size, subset_idxs, seed_size=30, scaler=None,
):
    """
    Computes the informativeness of each point within an active run.

    Parameters
    ----------
    idxs : np.array(M, N + seed_size)
        The indices of each array. Note that up to seed_size, these are ignored.
        It is from the start of seed size that they correlate to scores. M
        is the number of individual runs.

    scores : np.array(M, N)
        The scores associated with training a model. The i^th value of scores
        corresponds to a model trained on the first seed_size+(i-1)^th values
        idxs. M is the number of individual runs.

    dset_size : int, default=266
        The overall size of the dataset

    subset_idxs : np.array[int]
        The indices of the subset that the provided indices corresponds to.
        The reason this must be provided is quite simple. The idxs used in
        this calculation are not the original, but rather a subset selected
        by either the inner loop, or cross validation, or something similar.
        To correctly re-index them, the original indices they corresponded
        to must be provided. This can be turned off if None is passed, but
        this is strongly not recommended.

    seed_size : int, default=30
        The seed size that was selected, used to remove seed indices from the
        indice arrays

    Returns
    -------
    informativeness : np.ma.array(dset_size, results)
    """
    idxs, scores = ensure_array(idxs), ensure_array(scores)
    idxs, scores = np.atleast_2d(idxs), np.atleast_2d(scores)

    # Re-indexes the indices so that they match the provided subset.
    if subset_idxs is None:
        pass
    else:
        idxs = subset_idxs[idxs]

    informativeness = np.zeros((scores.shape[0], dset_size))

    for i, (idx, score) in enumerate(zip(idxs, scores)):
        informativeness[i] = index_scores(
            idx, score, seed_size=seed_size, dset_size=dset_size
        )

    return informativeness


@masked_values
def fold_informativeness(
    rep_results, strategy, runs, dset_size, scaler=None, seed_size=30, subset_idxs=None,
):
    """
    Wrapper function that computes the informativeness over all folds through
    multiple calls to informativeness_scoring. All parameters are the same,
    except the following

    rep_results {"Fold n" : {strategy : results}}
        The results to be passed

    strategy : str, default="Active"
        The sampling strategy that used used. Used to index the rep_results
        dictionary provided
    """
    informativeness = np.zeros((folds * runs, dset_size))

    for idx in range(folds):
        fold_values = informativeness_scoring(
            rep_results[strategy]["idxs"],
            rep_results[strategy]["scores"],
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
    results,
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
    Wrapper function that computes the informativeness over all sampling
    strategies of the data for a given representation.
    """
    informativeness_aggregate = []
    for idx, strategy in enumerate(strategies.keys()):
        result = informativeness_scoring(
            results[rep][strategy]["idxs"],
            results[rep][strategy]["scores"],
            seed_size=seed_size,
            scaler=scaler,
            dset_size=dset_size,
            subset_idxs=subset_idxs,
        )

        informativeness_aggregate.append(result)

    return np.concatenate(informativeness_aggregate)


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
    """
    Computes the informativeness values averaged over all representations and
    sampling strategies.
    """
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
