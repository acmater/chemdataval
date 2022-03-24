import numpy as np
import tqdm
from functools import partial


class Coarse_Graining:
    """
    Class that takes a dataset and performs the coarse graining algorithm on it.
    """

    def __init__(self, X, Y, training_idxs, testing_idxs, test_func):
        assert X.shape[0] == Y.shape[0], "X and Y must have the same first dim"
        self.X = X  # All X data to consider
        self.Y = Y  # All Y data to consider
        self.test_func = test_func  # function that will provide assessment
        self.scores = np.zeros((len(self.X),))  # array to store scores
        self.current_idxs = training_idxs
        self.testing_idxs = testing_idxs

    def __str__(self):
        return f"Current set: {self.current_idxs}, {self.scores[self.current_idxs]}"

    def coarse_graining(
        self, recursive_steps, C, runs, N, test_func=None, *args, **kwargs
    ):
        """
        Combines all other functionality to execute a coarse graining estimate
        of informativeness by running multiple coarse graining runs with
        an ever decreasing size of current indices.

        Parameters
        ----------
        recursive_steps : int
            The number of recursive steps to break the system down into.

        C : int
            The number of chunks to break the dataset into

        runs : int
            How many runs to do for each size

        N : int
            How many of the chunks to use during a single training run.

        test_func : <func>, default=None
            The test function used to score each stage of the coarse graining.

        Returns
        -------
        self.current_idxs : np.array
            The indices remaining following the coarse graining.

        self.scores[self.current_idxs] : np.array
            The scores associated with each index returned.
        """
        assert 2 ** recursive_steps <= (
            self.X.shape[0] // C
        ), f"""Too many recursive steps.
2**recursive_steps must be <= no. of datapoints * C.
Instead got {2** recursive_steps} for 2**recursive steps
and {self.X.shape[0] // C} for no. of datapoints / C."""

        for step in range(1, recursive_steps + 1):
            self.coarse_graining_run(
                C // step, runs, N // step, test_func, *args, **kwargs
            )

        return self.current_idxs, self.scores[self.current_idxs]

    def coarse_graining_run(self, C, runs, N, test_func=None, *args, **kwargs):
        """
        Performs a single step of coarse graining.

        Parameters
        ----------
        C : int
            How many chunks to separate the data into

        runs : int
            How many runs to do during training

        N : int < C
            How many of the chunks to use during a single training run.

        test_func : <func>, default=None
            The testing function to use during training.
        """
        assert N <= C, "N must be less than or equal to the number of chunks."
        chunks = self.subdivide_data(self.X[self.current_idxs], C)

        if test_func is None:
            test_func = self.test_func

        self.training_runs(chunks, runs, N, test_func=test_func, *args, **kwargs)
        self.select_next_set()
        return None

    def subdivide_data(self, X, C):
        """
        Takes input data X and subdivides it into N equal chunks that are
        randomly selected

        Parameters
        ----------
        X : np.array(N, M)
            The feature array to be split.

        C : int
            The number of chunks that X will be randomly split into.
        """
        assert C <= X.shape[0], "There must be more chunks than datapoints."
        perm = np.random.permutation(self.current_idxs)
        chunks = np.array_split(perm, C)
        return chunks

    def select_next_set(self, current_idxs=None):
        """
        Uses the current indices and scores to select the new set from the
        training data.
        """
        if current_idxs is None:
            current_idxs = self.current_idxs

        # Sort the scores at the current indices and reverse to get descending
        sorted = np.argsort(self.scores[current_idxs])[::-1]
        new_idxs = sorted[: len(sorted) // 2]

        # Override to create the new indices.
        # You need to re-index as you want the indices corresponding to the
        # current_idxs.
        self.current_idxs = self.current_idxs[new_idxs]
        return None

    def training_runs(self, chunks, runs, N, test_func=None, *args, **kwargs):
        """
        Repeats the evaluation a fixed number of times.

        Parameters
        ----------
        chunks : [np.array(X)]
            The indices that chunk the data arrays X and Y

        runs : int
            How many random runs to do.

        N : int < len(chunks)
            How many of the chunks to train this run on.

        test_func : <func>, default=None
            The testing function used to assess each chunk. If None, defaults
            to the self.test_func attribute.
        """
        for run in tqdm.tqdm(range(runs), position=0, leave=True):
            permutation = np.random.permutation(np.arange(len(chunks)))
            self.train_on_chunks(
                permutation, chunks, N, test_func=test_func, *args, **kwargs
            )
        return None

    def train_on_chunks(self, permutation, chunks, N, test_func=None, *args, **kwargs):
        """
        Evaluates a random permutation of the chunks on a given test function.

        Parameters
        ----------
        permutation : np.array(C,)
            A permutation of the chunks.

        chunks : [np.array(X)]
            The indices that chunk the data arrays X and Y

        N : int < len(chunks)
            How many of the chunks to train this run on.

        test_func : <func>, default=None
            The testing function used to assess each chunk. If None, defaults
            to the self.test_func attribute.
        """
        assert len(permutation) == len(
            chunks
        ), "The length of the permutation must equal how many chunks there are."

        if test_func is None:
            test_func = self.test_func

        assert test_func is not None, "A test function must be provided."
        assert N <= len(chunks), "N must be length that the number of chunks."

        chunks = np.array(chunks)
        scores = []
        for n in range(1, N + 1):
            chunk_idxs = permutation[:n]
            idxs = np.concatenate(
                chunks[chunk_idxs]
            )  # TODO Add function to ensure that you are not selecting testing indices.
            scores.append(
                test_func(self.X, self.Y, idxs, self.testing_idxs, *args, **kwargs)
            )
        # Compute how useful each chunk was.
        contributions = np.diff(scores)

        assert len(contributions) == len(
            chunks[1:N]
        ), "There must be one less contribution than chunks as the first chunk is the seed."
        # The reason for chunks[1:] is that you need to compare the model to
        # one without the datapoint. You can't compare the model to one
        # with no data, so the first chunk is always taken as seed data.
        self.assign_performance_to_chunk(zip(contributions, chunks[permutation[1:N]]))
        return None

    def assign_performance_to_chunk(self, performances):
        """
        Takes the performance output which assigns an informativeness value to
        each chunk and assigns the average value from the chunk across its
        inputs.

        Parameters
        ----------
        performances : [(performance, chunk)]
        """
        for (performance, chunk) in performances:
            self.scores[chunk] += performance / len(chunk)


if __name__ == "__main__":
    test_X = np.random.randn(16, 5)
    test_Y = np.random.randn(16,)

    def random_test_func(X, Y, n=1):
        return np.random.randn() * n

    test = Coarse_Graining(test_X, test_Y, random_test_func)


    # print(
    #    test.train_on_chunks(
    #        np.random.permutation(np.arange(len(chunks))),
    #        chunks,
    #        len(chunks),
    #        partial(random_test_func, n=1),
    #    ),
    # )
    # print(test.scores)
    # print(test.select_next_set(None))
    # print(test)

    # chunks = test.subdivide_data(test_X[test.current_idxs], 4)
    # print(test.training_runs(chunks, 10, len(chunks), partial(random_test_func, n=1),))
    # print(test.scores)
    # print(test.select_next_set(None))
    # print(test)

    # print(test)
    # test.coarse_graining_run(8, 10, 4)
    # print(test)
    # test.coarse_graining_run(4, 10, 2)
    # print(test)
    # test.coarse_graining_run(2, 10, 1)
    # print(test)

    print(test.coarse_graining(3, 8, 10, 4))
