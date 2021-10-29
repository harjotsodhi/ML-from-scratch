import numpy as np


def one_hot_encoder(arr):
    """
    Transforms an (m x 1) array consisting of k unique categorical variables
        to an (m x k) array where each column k_i is a binary response
        variable w.r.t to the unique categorical variable k_i

    Parameters
    ----------
    arr: np.array
        vector (m x 1)
    """
    # check the number of classes to tell whether one-hot-encoding is needed
    n_unique = np.unique(arr, return_counts=False)
    k = n_unique.shape[0]

    if k > 2:
        # initialize a (m x k) matrix
        mat = np.zeros((arr.shape[0], k))
        for c in range(k):
            mask = (arr[:,0]==n_unique[c])
            mat[mask,c] = 1.

        return mat

    # if the number of classes is <=2 , then one-hot-encoding is not needed
    else:
        return arr


def sigmoid(arr):
    """
    Translate the input array into the range [0,1] through the
        sigmoid activation function.

    Parameters
    ----------
    arr: np.array
        vector (m x 1)
    """
    return 1/(1+np.exp(-arr))


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


def standardize(X):
    """
    Subtract the mean and scale to unit variance.

    Parameters
    ----------
    X: np.array
        matrix (m x n)
    """
    X_centered = X - np.mean(X, axis=0)
    X_scaled = X_centered/np.std(X_centered, axis=0)
    return X_scaled


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
