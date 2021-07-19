import numpy as np
from ..helpers import helpers as hp


class Linear_regression(object):
    """
    Simple and multiple linear regression model.
    """
    def __init__(self, method="gradient_descent"):
        if method not in ("least_squares", "gradient_descent"):
            raise ValueError('Method param must be "least_squares" or "gradient_descent"')

        self.method = method
        self.fit_called = False


    def ols(self, X, y):
        '''
        Fit model using ordinary least squares method
        Minimize SSE: argmin_b { ((y - (X @ b)).T) @ ((y - (X @ b))) }

        b = inverse(X.T @ X) @ X.T @ y
        '''
        coef = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return coef


    def fit(self, X, y, normalize=False):

        if self.fit_called:
            raise ValueError('Fit method already called')

        self.fit_called = True
        # normalize
        if normalize:
            X = hp.normalize(X)

        # format np.arrays for regression
        X,y = hp.format_reg(X, y, normalize)

        if self.method == "least_squares":

            # fit through OLS
            coef = self.ols(X, y)

        else:
            pass

        # clean up
        if not normalize:
            self.intercept = coef[0]
            self.coef = coef[1:].T
        else:
            self.intercept = 0
            self.coef = coef.T


    def get_coef(self):
        '''
        Return the fitted coefficients
        '''

        if not self.fit_called:
            raise ValueError('Model not yet fit')

        return self.coef.flatten()


    def get_intercept(self):
        '''
        Return the fitted intercept
        '''

        if not self.fit_called:
            raise ValueError('Model not yet fit')

        return self.intercept.flatten()
