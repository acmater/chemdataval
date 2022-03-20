import numpy as np

import unittest
import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations

import chemdataval

from chemdataval.preprocessing import standardise
from chemdataval.utils import data_stats, kmax, kindices, normalise
from chemdataval.informativeness import (
    informativeness_scoring,
    masked_values,
    index_scores,
)
from chemdataval.query_strategy import modify_std
from chemdataval.testing_functions import fold_testing, active_test, random_test
from chemdataval.explicit_shapley import Shapley_Values

test_X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
test_Y = np.array([3, 4, 5])


class TestPreprocessing(unittest.TestCase):
    def test_standardise(self):
        X, Y = standardise(test_X, test_Y, train_index=[0, 1, 2])
        assert np.all(
            X.mean(axis=0) == np.zeros((3,))
        ), "The standardisation is not working correctly."
        assert Y.mean(axis=0) == 0, "Standardisation is not working correctly."

    def test_empty_standardise(self):
        with self.assertRaises(Exception):
            X = standardise([])

    def test_standardise_train_only(self):
        X, Y = standardise(test_X, test_Y, train_index=[0, 1])
        assert ~np.all(
            X.mean(axis=0) == np.zeros((3,))
        ), "Standardisation should not have the average as zero as it only uses some of the training data."
        assert ~(
            Y.mean(axis=0) == 0
        ), "Standardisation should not set the average Y value to zero."


class TestUtilities(unittest.TestCase):
    def test_datastats(self):
        μ, σ = data_stats(test_X)
        assert np.all(
            μ == np.array([4, 5, 6])
        ), "Data Stats function is not producing the correct mean."
        assert np.all(
            σ == np.sqrt(np.var(test_X, axis=0))
        ), "Data Stats function is not producing correct standard deviation."

    def test_kmax(self):
        vals = kmax(np.array([1, 3, 2, 5, 6, 7]), k=3, indices=False, sort=True)
        assert len(vals) == 3, "There are not the correct number of indices for kmax"
        assert np.all(
            vals == np.array([5, 6, 7])
        ), "The values returned by kmax are incorrect."

    def test_kindices(self):
        res = kindices(idxs=np.array([0, 2, 1]), arr=np.array([1, 3, 2]), indices=True)
        assert np.all(
            res == np.array([0, 2, 1])
        ), "kindices is not returning the correct indices."
        res_sorted = kindices(
            np.array([0, 2, 1]), arr=np.array([1, 3, 2]), indices=False, sort=True
        )
        assert np.all(
            res_sorted == np.array([1, 2, 3])
        ), "kindices with sorting is not returning the correct array."
        res_unsorted = kindices(
            np.array([0, 2, 1]), arr=np.array([1, 3, 2]), indices=False, sort=False
        )
        assert np.all(
            res_unsorted == np.array([1, 2, 3])
        ), "Unsorted kindices is not working correctly."

    def test_normalise(self):
        normed = normalise(test_Y)
        assert np.allclose(
            np.linalg.norm(normed), 1
        ), "normalise is not correctly normalising a vector."


class TestInformativeness(unittest.TestCase):
    def test_masking_decorator(self):
        @masked_values
        def test_func():
            return np.array([1, 2, 0, 3, 4])

        assert np.all(
            test_func() == np.ma.masked_equal(np.array([1, 2, 0, 3, 4]), 0)
        ), "Masking operation not working correctly."

    def test_index_scoring(self):
        idxs = np.array([1, 2, 3, 4, 5, 6])
        scores = np.array([0.1, 0.3, 0.6, 0.5])
        desired = np.ma.masked_equal(np.array([0, 0, 0, 0, 0.2, 0.3, -0.1]), 0)
        assert np.allclose(
            desired, index_scores(idxs, scores, seed_size=3, dset_size=7)
        ), "index_scores is not working correctly."

    def test_informativeness_scoring(self):
        idxs = np.array([[1, 2, 3, 4, 5, 6], [3, 6, 4, 1, 0, 5]])
        scores = np.array([[0.1, 0.3, 0.6, 0.5], [0.2, 0.4, 0.7, 0.8]])
        desired = np.ma.masked_equal(
            np.array([[0, 0, 0, 0, 0.2, 0.3, -0.1], [0.3, 0.2, 0, 0, 0, 0.1, 0]]), 0
        )
        print(
            (
                desired,
                informativeness_scoring(
                    idxs, scores, seed_size=3, dset_size=7, subset_idxs=None
                ),
            )
        )
        assert np.ma.allclose(
            desired,
            informativeness_scoring(
                idxs, scores, seed_size=3, dset_size=7, subset_idxs=None
            ),
        ), "informativeness_scoring is not working correctly."


class TestQuery(unittest.TestCase):

    query_test_X = np.array(
        [
            [-1.572, 0.297],
            [0.916, 1.123],
            [1.071, 0.285],
            [-1.678, -2.020],
            [-0.843, -0.097],
            [-0.380, 0.501],
            [0.879, -0.985],
            [-2.304, 0.690],
            [1.172, 1.415],
            [0.048, -0.016],
        ]
    )

    query_test_informativeness = np.array(
        [0.863, 1.148, 0.053, 1.151, 1.587, 0.069, 0.715, 1.692, 0.428, 1.395]
    )
    similarity = 1 / (1 + cdist(query_test_X, query_test_X, metric="euclidean"))

    def test_high_beta(self):
        # High β value, similarity should dominate.
        assert (
            np.argmax(
                modify_std(
                    self.query_test_informativeness,
                    self.similarity.sum(axis=0),
                    50,
                    np.arange(10),
                    np.arange(10),
                    False,
                )
            )
            == 9
        ), "The modify_std method is not working for β=50 is not working."

    def test_zero_beta(self):
        # β = 0, so it should just use uncertainty.
        assert (
            np.argmax(
                modify_std(
                    self.query_test_informativeness,
                    self.similarity.sum(axis=0),
                    0,
                    np.arange(10),
                    np.arange(10),
                    False,
                )
            )
            == 7
        ), "The modify_std method is not working for β=zero is not working."

    def test_zero_one(self):
        # Balanced sampling, should use the maximum combination of the two.
        assert (
            np.argmax(
                modify_std(
                    self.query_test_informativeness,
                    self.similarity.sum(axis=0),
                    1,
                    np.arange(10),
                    np.arange(10),
                    False,
                )
            )
            == 4
        ), "The modify_std method is not working for β=1 is not working."

    def test_large_batch(self):
        # Test with a larger batch size
        assert np.all(
            np.isin(
                kmax(
                    modify_std(
                        self.query_test_informativeness,
                        self.similarity.sum(axis=0),
                        1,
                        np.arange(10),
                        np.arange(10),
                        False,
                    ),
                    3,
                    indices=True,
                ),
                np.array([7, 9, 4]),
            )
        ), "The modify_std method fails with a large batch size."


class TestShapley(unittest.TestCase):
    options = [0, 1, 2]
    combinations_ = [()]
    for i in range(1, len(options) + 1):
        combinations_.extend(list(combinations(options, i)))

    test_example = {
        key: val for key, val in zip(combinations_, [50, 40, 48, 100, 39, 85, 95, 83])
    }

    DS = Shapley_Values(test_example)

    def test_value(self):
        assert np.allclose(
            self.DS.Shapley_Value(2), 46.6666
        ), "Data Shapley computation is not working correctly."


if __name__ == "__main__":
    unittest.main()
