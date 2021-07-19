import numpy as np


def format_reg(X, y, normalized):
    '''
    Format np.arrays for regression
    '''
    # add column vector of ones's for the intercept term
    if not normalized:
        try:
            X[:,0] == np.ones(X.shape[0])
        except:
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

    # reshape y, for matrix algebra
    try:
        y.shape[1] == 1
    except:
        y = y.reshape(y.shape[0],1)

    return X, y


def normalize(X):
    '''
    Normalize X matrix
    '''
    pass
