"""
Module contains the function to intelligently sample given a set of data
informativeness values.
"""

import tqdm
import warnings
import numpy as np
from .utils import softmax, check_invalid_selection


def intelligent_sampling(
    test_func,
    informativeness,
    X,
    Y,
    train_idxs,
    test_idxs,
    sizes,
    repeats=10,
    T=10000,
    control=False,
    *args,
    **kwargs
):
    """
    Function to take a machine learning model and pre-computed informativeness
    values and assess their performance as the dataset grows in size.

    Parameters
    ----------
    test_func : <func>
        A function that will test the performance of the intelligently selected
        training data points.

    informativeness : np.array
        The informativeness of each point in the training data computed

    X : np.array
        The features of the samples to be trained on

    Y : np.array
        The ground truth vector

    train_idxs : np.array
        The indices of the training data - these must perfectly correlate to
        the provided informativeness array

    test_idxs : np.array
        The indices of the testing set

    repeats : int, default=10
        The number of times the sampling step is repeated

    representations : dict, default=representations
        The representations of the molecules provided with a label
        in dictionary form

    sizes : iter
        The sizes that will be iterated over and used to train the
        sklearn regressor

    T : float, default=10,000
        The temperature term that will scale the probability vector
    """
    if control is True:
        # Sets all probabilities to uniform so that it becomes random
        informativeness = np.ones(informativeness.shape)
        # Override temperature parameter
        T = 1

    with warnings.catch_warnings():
        # Remove the convergence and nearing bound warnings that fill up
        # the output section
        warnings.simplefilter("ignore")

        current_informativeness = informativeness * T

        # Occasionally there is a glitch in the kf system which leaves a
        # training point never selected, this fills its values with zero.
        if np.ma.isMaskedArray(current_informativeness):
            current_informativeness = current_informativeness.filled(
                0
            )  # Line because 134 is exactly equal to zero.

        results = []

        for size in tqdm.tqdm(sizes):
            results_repeats = []

            for _ in range(repeats):
                selected_idxs = np.random.choice(
                    train_idxs,
                    size=(int(size),),
                    replace=False,
                    # Select only the relevant values of informativeness
                    p=softmax(current_informativeness[train_idxs]),
                )

                check_invalid_selection(selected_idxs, test_idxs)
                results_repeats.append(
                    test_func(X, Y, selected_idxs, test_idxs, *args, **kwargs)
                )
            results.append(results_repeats)

        return results


def intelligent_sampling_over_representations(
    test_func,
    informativeness_dict,
    representations,
    Y,
    train_idxs,
    test_idxs,
    sizes,
    repeats=10,
    skip_reps=[],
    T=10000,
    control=False,
    dset_size=None,
    *args,
    **kwargs
):
    """
    Performs intelligent sampling over a set of representations.

    Parameters
    ----------
    informativeness_dict : np.array
        The informativeness array for each key in representations

    test_func : <func>
        A function that will test the performance of the intelligently selected
        training data points.

    X : np.array
        The features of the samples to be trained on

    Y : np.array
        The ground truth vector

    train_idxs : np.array
        The indices of the training data - these must perfectly correlate
        to the provided informativeness array

    test_idxs : np.array
        The indices of the testing set

    repeats : int, default=10
        The number of times the sampling step is repeated

    representations : dict, default=representations
        The representations of the molecules provided with a label
        in dictionary form

    sizes : iter
        The sizes that will be iterated over and used to train the
        sklearn regressor

    T : float, default=10,000
        The temperature term that will scale the probability vector
    """

    results = {}

    assert (
        representations.keys() == informativeness_dict.keys()
    ), "The keys in representations and informativeness_dict must be the same."

    for rep in representations.keys():
        if rep in skip_reps:
            continue

        X = representations[rep]
        informativeness = informativeness_dict[rep]
        assert (
            len(informativeness.shape) == 1
        ), "informativeness must be 1D at this point."

        results[rep] = intelligent_sampling(
            test_func,
            informativeness,
            X,
            Y,
            train_idxs,
            test_idxs,
            sizes,
            repeats=repeats,
            T=T,
            control=control,
            *args,
            **kwargs
        )
    return results
