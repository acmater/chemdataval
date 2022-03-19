from .utils import test_model, check_invalid_selection


def test_permutation(
    model, X, Y, train_idxs, test_idxs, queries=None, seed_size=30, *args, **kwargs
):
    """Simple function to take a particular model with both inputs and outputs,
    and get the performance of the model using each of the possible queries.

    Parameters
    ----------
    model : sklearn.RegressorMixin
        The model to be tested.

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
        The seed_size that will be used to select a portion of the train indices
        to train the first model.

    *args, **kwargs
        Optional arguments that are passed to the model at instantiation.

    Returns
    -------
    results : List
        The list of results.
    """
    assert train_idxs < len(Y), "Train indices is too long."
    assert test_idxs < len(Y), " Test indices is too long."
    assert queries < (len(train_idxs) - seed_size), "Too many queries."
    check_invalid_selection(train_idxs, test_idxs)

    if queries is None:
        queries = train_idxs - seed_size

    scores = np.zeros((queries + 1,))

    # The plus one is needed so that the last of the queries is included, as
    # the first one is just the seed dataset.
    for idx, num_queries in enumerate(range(queries + 1)):
        train_data = train_idxs[: seed_size + num_queries]
        scores[idx] = test_model(model, X, Y, train_data, test_idxs, *args, **kwargs)

    return results
