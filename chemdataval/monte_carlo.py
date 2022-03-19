from .utils import check_invalid_selection
from .preprocessing import preprocess
from .informativeness import informativeness_scoring
import numpy as np
import warnings
import tqdm


def test_model(model, X, Y, train_idxs, test_idxs, *args, **kwargs):
    """
    Generic model testing function


    Parameters
    ----------
    model : sklearn.MixinRegressor

    """
    check_invalid_selection(train_idxs, test_idxs)

    X, Y = preprocess(X, Y, train_idxs)
    model = model(*args, **kwargs)
    model.fit(X[train_idxs], Y[train_idxs])
    return model.score(X[test_idxs], Y[test_idxs])


def test_permutation(
    test_func, X, Y, train_idxs, test_idxs, queries=None, seed_size=30, *args, **kwargs
):
    """Simple function to take a particular model with both inputs and outputs,
    and get the performance of the model using each of the possible queries.

    Parameters
    ----------
    test_func : <func>
        The function used to test each stage of the permutation. The idea here
        is that you can plugin different functions such as machine learning
        models or any other system of interest.

    X : np.array(N,M)
        The covariates to regress on. N is the number of samples, and M the
        dimension.

    Y : np.array(N,)
        The ground truth labels for each input.

    train_idxs : np.array(<=N,)
        The training indices to be used.

    test_idxs : np.array(<=N)
        The testing indices to be used.

    queries : int <= len(train_index) - seed_size, default=None
        The number of queries. If None, defaults to all possible points in the
        train_index minus the seed_size

    seed_size : int, default=30
        The seed_size that will be used to select a portion of the train
        indices to train the first model.

    *args, **kwargs
        Optional arguments that are passed to the model at instantiation.

    Returns
    -------
    results : List
        The list of results.
    """
    assert len(train_idxs) < len(Y), "Train indices is too long."
    assert len(test_idxs) < len(Y), " Test indices is too long."
    assert queries <= (len(train_idxs) - seed_size), "Too many queries."
    check_invalid_selection(train_idxs, test_idxs)

    if queries is None:
        queries = train_idxs - seed_size

    scores = np.zeros((queries + 1,))

    # The plus one is needed so that the last of the queries is included, as
    # the first one is just the seed dataset.

    for idx, num_queries in enumerate(range(queries + 1)):
        train_data = train_idxs[: seed_size + num_queries]
        scores[idx] = test_func(
            X=X, Y=Y, train_idxs=train_data, test_idxs=test_idxs, *args, **kwargs
        )

    return scores


def test_inner_folds(
    test_func,
    kf,
    X,
    Y,
    seed_size,
    inner_idxs,
    outer_idxs,
    groups=None,
    queries=None,
    random_runs=100,
    *args,
    **kwargs,
):
    """
    Function bundles up other functions and allows the user to take a full dataset
    and provide the inner and outer idxs. This system will then use the kf
    to split the inner_idxs, and then deploy the permutation testing and
    provided test function to assess the performance.

    Parameters
    ----------
    test_func : <func>
        The function used to test each stage of the permutation. The idea here
        is that you can plugin different functions such as machine learning
        models or any other system of interest.

    kf : sklearn.ShuffleSplit
        The splitter used to separate the inner_idxs

    X : np.array(N,M)
        The covariates to regress on. N is the number of samples, and M the
        dimension.

    Y : np.array(N,)
        The ground truth labels for each input.

    inner_idxs : np.array
        The inner indices that the model is going to perform k fold cross validation
        on.

    outer_idxs : np.array
        The outer indices that the models are never allowed to see. These are given
        to ensure that there is never any overlap

    groups : np.array(N)
        The labels used to split the categorical shuffler, if a stratified
        splitter is provided.

    random_runs : int, default=100
        The number of random permutation generation runs that the system will conduct.

    *args, **kwargs
        Given to the test function method provided.
    """
    if groups is None:
        groups = np.zeros((len(inner_idxs) + len(outer_idxs),))

    with warnings.catch_warnings():
        # Remove the convergence and nearing bound warnings that fill up the output section
        warnings.simplefilter("ignore")

        idxs, scores = [], []

        for kf_idx, item in enumerate(kf.split(X[inner_idxs], y=groups[inner_idxs])):
            # When going into this loop, the train and test indices are subsets of the
            # length of inner_idxs. This means that to point to the correct datapoints
            # at a higher leve, they have to be re-indexed.

            print(f"Running fold {kf_idx+1}")
            train_index, test_index = item
            # Re-indexing the values
            train_index, test_index = (
                inner_idxs[train_index],
                inner_idxs[test_index],
            )

            if queries is None:
                queries = len(train_index) - seed_size

            X, Y = preprocess(X, Y, train_index)
            for run in tqdm.tqdm(range(random_runs), position=0, leave=True):
                # Generate the permutation
                perm = np.random.permutation(train_index)

                # Ensure that the outer mols are never appearing in the train or test index.
                assert (
                    len(
                        np.intersect1d(
                            np.concatenate([train_index, test_index]), outer_idxs
                        )
                    )
                    == 0
                ), "The training and test indices are overlapping with the outer mols."

                # Generate the scores
                run_scores = test_permutation(
                    test_func,
                    X,
                    Y,
                    perm,
                    test_index,
                    queries=queries,
                    seed_size=seed_size,
                    *args,
                    **kwargs,
                )

                idxs.append(perm[: seed_size + queries])
                scores.append(run_scores)

        test_inform = informativeness_scoring(
            idxs,
            scores,
            dset_size=len(inner_idxs) + len(outer_idxs),
            seed_size=seed_size,
        ).mean(axis=0)

        # Check that no point in the outermols set is being given an informativeness value.
        assert (
            len(np.intersect1d(np.where(~test_inform.mask), outer_idxs)) == 0
        ), "Points in the outer mols set are being assigned an informativeness value."
        return idxs, scores
