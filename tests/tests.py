import numpy as np

import unittest
import numpy as np 

from chemdataval.preprocessing import * 
from chemdataval.utils import *

test_X = np.array([[1,2,3],[4,5,6],[7,8,9]])
test_Y = np.array([3,4,5])

class TestPreprocessing(unittest.TestCase):
    def test_standardise(self):
        X, Y = standardise(test_X, test_Y, train_index=[0,1,2])
        assert np.all(X.mean(axis=0) == np.zeros((3,))), "The standardisation is not working correctly."
        assert Y.mean(axis=0) == 0, "Standardisation is not working correctly."

    def test_standardise_train_only(self):
        X, Y = standardise(test_X, test_Y, train_index=[0,1])
        assert ~np.all(X.mean(axis=0) == np.zeros((3,))), "Standardisation should not have the average as zero as it only uses some of the training data."
        assert ~(Y.mean(axis=0) == 0), "Standardisation should not set the average Y value to zero."


class TestUtilities(unittest.TestCase):
    def test_datastats(self):
       μ, σ = data_stats(test_X)
       assert np.all(μ == np.array([4,5,6])), "Data Stats function is not producing the correct mean."
       assert np.all(σ == np.sqrt(np.var(test_X, axis=0))), "Data Stats function is not producing correct standard deviation."

    def test_kmax(self):
        vals = kmax(np.array([1,3,2,5,6,7]), k=3, indices=False, sort=True)
        assert len(vals) == 3, "There are not the correct number of indices for kmax"
        assert np.all(vals == np.array([5,6,7])), "The values returned by kmax are incorrect."
    
    
    def testkindices(self):
        res = kindices(idxs=np.array([0,2,1]), arr=np.array([1,3,2]), indices=True)
        assert np.all(res == np.array([0,2,1])), "kindices is not returning the correct indices."
        res_sorted = kindices(np.array([0,2,1]), arr=np.array([1,3,2]), indices=False, sort=True)
        assert np.all(res_sorted == np.array([1,2,3])), "kindices with sorting is not returning the correct array."
        res_unsorted = kindices(np.array([0,2,1]), arr=np.array([1,3,2]), indices=False, sort=False)
        assert np.all(res_unsorted == np.array([1,2,3])), "Unsorted kindices is not working correctly."


    
if __name__ == "__main__":
        unittest.main()
