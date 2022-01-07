import numpy as np
from sklearn import decomposition
from sklearn.datasets import load_iris
from mlfromscratch.unsupervised.pca import PCA
from mlfromscratch.helpers import helpers as hp


def test():

    iris = load_iris()
    X = iris.data

    pca = PCA(n_components=2)
    pca.fit(X)
    X_reduced = pca.project(X)

    pca_sklearn = decomposition.PCA(n_components=2)
    X = hp.standardize(X)
    pca_sklearn.fit(X)
    X_reduced_sklearn = pca_sklearn.transform(X)

    # check if projections are equal
    # the sign between sklearn's projections may be swapped, so just ignore
    np.testing.assert_allclose(abs(X_reduced), abs(X_reduced_sklearn))


if __name__ == '__main__':
    test()
