import numpy as np



class K_means:
    def __init__(self, num_clusters, seed=42):
        """
        Implements the K-means clustering algorithm

        Parameters
        ----------
        num_clusters: int
            the number of clusters, k.

        seed: int, default=42
            controls the randomness of initial centroids
        """
        self.k = num_clusters
        self.seed = seed


    def fit(self, X):
        """
        Iteratively fit k clusters until convergence.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)
        """
        # randomly initialize k centroids
        centroids = self._initial_centroids(X)
        # record initial centroids for convergence check
        self.previous_centroids = centroids
        # convergence indicator
        converged = False

        while not converged:
            dist_mat = self._dist(X, centroids)
            clusters = self._assign(dist_mat)
            centroids = self._update_centroids(X, clusters)
            converged = self._check_convergence(centroids)
            self.previous_centroids = centroids

        # record the final centroids
        self.centroids = centroids


    def predict(self, X):
        '''
        Return the predicted value.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)
        '''
        dist_mat = self._dist(X, self.centroids)
        y_pred = self._assign(dist_mat)
        return y_pred


    ### Private Methods ###

    def _initial_centroids(self, X):
        """
        Initialize k centroids.

        Randomly select (without replacement) k initial
            centroids.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)
        """
        np.random.seed(self.seed)
        ind = np.random.choice(X.shape[0], size=self.k, replace=False)
        centroids = X[ind, :]
        return centroids


    def _dist(self, X, centroids):
        """
        Calculate the Euclidean distance between each point
            and the K centroids.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        centroids: np.array
            centroids matrix (k x n)
        """
        dist_mat = np.zeros((X.shape[0], self.k))

        for r in range(X.shape[0]):
            for c in range(self.k):
                dist_mat[r,c] = sum((X[r,:]-centroids[c,:])**2)**(1/2)

        return dist_mat


    def _assign(self, dist_mat):
        """
        Assign each data point to the cluster of its
            nearest centroid.

        Parameters
        ----------
        dist_mat: np.array
            distance matrix (m x k)
        """
        clusters = np.argmin(dist_mat, axis=1)
        return clusters


    def _update_centroids(self, X, clusters):
        """
        Update centroids to be the mean of each cluster

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        centroids: np.array
            centroids matrix (k x n)
        """
        centroids = np.zeros((self.k,X.shape[1]))

        for c in range(self.k):
            centroids[c] = X[clusters==c].mean(axis=0)

        return centroids


    def _check_convergence(self, centroids):
        """
        Test for convergence.

        Whether the centroids have changed.

        Parameters
        ----------
        centroids: np.array
            centroids matrix (k x n)
        """
        return np.array_equal(centroids, self.previous_centroids)
