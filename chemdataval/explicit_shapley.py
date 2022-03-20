import numpy as np
from itertools import combinations
import math
from scipy.special import binom


class Shapley_Values:
    """
    A class that takes a dataset and provides the methods to compute the
    explicit shapley values.

    Code inspired and example taken from the following page:
    https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30

    Parameters
    ----------
    """

    def __init__(self, data):
        assert isinstance(data, dict), "Data must be provided as a dict"
        self.data = data

        assert math.log2(
            len(self.data)
        ).is_integer(), "Data must be a powerset and thus have 2^N elements"
        self.F = int(math.log2(len(self.data)))

    @staticmethod
    def generate_datapoint_without_foi(datapoint, feature_of_interest):
        """
        Generates a new tuple with the feature of interest removed.
        """
        new = []
        for val in datapoint:
            if val == feature_of_interest:
                continue
            else:
                new.append(val)
        return new

    def MC_value(self, datapoint, feature_of_interest):
        """
        Computes a since MC value for a feature or dataset.
        """
        assert (
            feature_of_interest in datapoint
        ), "The feature of interest must be in the datapoint."

        new = tuple(self.generate_datapoint_without_foi(datapoint, feature_of_interest))

        return self.data[datapoint] - self.data[new]

    def Shapley_Value(self, feature_of_interest):
        """
        Computes an individual shapley value for a given feature of interest.

        Parameters
        ----------
        feature_of_interest : int
            The integer identifier of the feature of interest.
        """
        datapoints_with_foi = []

        for datapoint in self.data.keys():
            if feature_of_interest in datapoint:
                datapoints_with_foi.append(datapoint)

        Shapley = 0
        for datapoint in datapoints_with_foi:
            f = len(datapoint)
            Shapley += np.reciprocal(f * binom(self.F, f)) * self.MC_value(
                datapoint, feature_of_interest
            )

        return Shapley

    def Shapley_Values(self):
        """
        Computes the Shapley values for all features and returns them as a
        dictionary.
        """
        shapley_vals = [self.Shapley_Value(f) for f in range(self.F)]
        assert np.allclose(
            np.sum(shapley_vals), self.data[tuple(range(self.F))] - self.data[()]
        ), "The Shapley values are not summing to the value when all features are present."

        return {f: val for f, val in zip(range(self.F), shapley_vals)}


if __name__ == "__main__":

    options = [0, 1, 2]
    combinations_ = [()]
    for i in range(1, len(options) + 1):
        combinations_.extend(list(combinations(options, i)))

    test_example = {
        key: val for key, val in zip(combinations_, [50, 40, 48, 100, 39, 85, 95, 83])
    }

    DS = Shapley_Values()(test_example)

    print(DS.MC_value((0,), 0))
    print(DS.Shapley_Value(2))
