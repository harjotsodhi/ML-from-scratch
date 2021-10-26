import numpy as np
from ..helpers import helpers as hp



class PCA(object):

    def __init__(self, n_components=2):
        '''
        Implements the PCA algorithm.

        Parameters
        ----------
        n_components: Number of principal components
        '''
        self.n_components = n_components


    def fit(self, X):
        '''
        Calculate the first n_components eigenvectors

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)
        '''
        X_scaled = hp.standardize(X)
        cov_mat = self._covariance(X_scaled)
        eig_vals, eig_vects = self._eigen_decomp(cov_mat)
        # sort the eigenvectors by their respective eigenvalues
        order = np.argsort(eig_vals)[::-1]
        self.eig_vects = eig_vects[:,order]


    def project(self, X):
        '''
        Project data on to n_components principal components

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)
        '''
        X_scaled = hp.standardize(X)
        x_project = X_scaled.dot(self.eig_vects[:,:self.n_components])
        return x_project


    ## private methods ##


    def _covariance(self, X):
        '''
        Compute the covariance (n x n) matrix.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)
        '''
        return np.cov(X.T)


    def _eigen_decomp(self, X):
        '''
        Decomposes a matrix into eigenvectors and eigenvalues.

        Parameters
        ----------
        X: np.array
            feature matrix (n x n)
        '''
        assert X.shape[0] == X.shape[1], "X must be a square matrix"
        eig_vals, eig_vects = np.linalg.eig(X)
        return eig_vals, eig_vects
