import numpy as np


def mode(arr):
    """
    Calculate the mode.

    Parameters
    ----------
    arr: np.array
        vector (m x 1)
    """
    vals,counts = np.unique(arr, return_counts=True)
    index = np.argmax(counts)
    return arr[index]


def single_class(arr):
    """
    Check whether all values are same.

    Parameters
    ----------
    arr: np.array
        vector (m x 1)
    """
    return np.all(arr == arr[0])


def format_reg(X, y=None, normalized=False):
    '''
    Format np.arrays for regression.
    '''
    # add column vector of ones's for the intercept term
    if not normalized:
        try:
            if not all(X[:,0] == np.ones(X.shape[0])):
                raise ValueError

        except ValueError:
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

    # reshape y, for matrix algebra, if y provided
    if y is None:
        return X
    else:
        try:
            y.shape[1] == 1
        except:
            y = y.reshape(-1,1)

        return X, y


def normalize(X):
    '''
    Normalize X matrix.
    '''
    pass
