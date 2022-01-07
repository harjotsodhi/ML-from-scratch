import numpy as np
from mlfromscratch.helpers import helpers as hp
from mlfromscratch.cart import Classification_tree



class Random_forest(object):
    """
    Implements the Random Forest ensemble method for training
        "num_trees" decision trees on sub-samples of the data.

    Parameters
    ----------
    num_trees: int, default=10
        number of trees in the forest.

    num_samples: int, default=None
        number of samples to draw (with replacement)
        for each bootstrapped dataset.
        If None then all samples are used.

    seed: int, default=17
        controls the randomness of the bootstrapping.
    """
    def __init__(self, num_trees=10, num_samples=None, seed=17):
        self.trees = [Classification_tree() for n in range(num_trees)]
        self.num_samples = num_samples
        self.seed = seed


    def fit(self, X, y):
        """
        Fit "num_trees" classification trees using
            bootstrapped samples of size "num_samples"

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        y: np.array
            response vector (m x 1)
        """
        for t in self.trees:
            bootstrap_ind = self._bootstrap(X)
            t.fit(X[bootstrap_ind], y[bootstrap_ind])


    def predict(self, X):
        '''
        Return the predicted value.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)
        '''
        y_pred_all = np.zeros((X.shape[0], len(self.trees)))
        y_pred = np.zeros(X.shape[0])

        for c in range(len(self.trees)):
            y_pred_all[:, c] = self.trees[c].predict(X)

        for r in range(X.shape[0]):
            y_pred[r] = hp.mode(y_pred_all[r,:])

        return y_pred


    ### Private Methods ###

    def _bootstrap(self, X):
        '''
        Random sampling of feature matrix X
            with replacement

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)
        '''
        np.random.seed(self.seed)

        if not self.num_samples:
            self.num_samples = X.shape[0]

        ind = np.random.choice(X.shape[0], self.num_samples, replace=True)
        return ind
