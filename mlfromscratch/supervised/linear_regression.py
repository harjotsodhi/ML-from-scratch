import numpy as np
from ..solvers.gradient_descent import Gradient_descent
from ..helpers import helpers as hp



class Linear_regression(object):
    """
    Simple and multiple linear regression model.

    Parameters
    ----------
    method: str, default='gradient_descent'
        method for deriving parameters, 'least_squares' or 'gradient_descent'
    """
    def __init__(self, method="gradient_descent"):
        if method not in ("least_squares", "gradient_descent"):
            raise ValueError('Method param must be "least_squares" or "gradient_descent"')

        self.method = method
        self.fit_called = False


    def ols(self, X, y):
        '''
        Fit model using the ordinary least squares method
        Minimize SSE: argmin_b { ((y - (X @ b)).T) @ ((y - (X @ b))) }

        b = inverse(X.T @ X) @ X.T @ y

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        y: np.array
            response vector (m x 1)
        '''
        coef = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return coef


    def gd(self, X, y):
        '''
        Fit model using the gradient descent method.
        Encode cost and gradient as functions.

        cost = MSE => SSE/2m => ((y_pred - y).T @ (y_pred - y))/2m
        gradient = partial derivative of MSE wrt coefs => (X.T @ (y_pred - y))/m

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        y: np.array
            response vector (m x 1)
        '''
        # MSE
        def cost(m, err): return (1/(2*m)) * err.T.dot(err)
        # gradient (partial derivative of MSE wrt coefs)
        def gradient(m, X, err): return (1/m)*(X.T.dot(err))
        # solve
        solver = Gradient_descent(gradient, cost, max_iter=10000, abs_tol=1e-9)
        coef = solver.solve(X, y, learning_rate=0.01)
        return coef


    def fit(self, X, y, normalize=False, learning_rate=0.01):
        '''
        Estimate the model parameters using the specified method.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        y: np.array
            response vector (m x 1)

        normalize: bool, default='False'
            whether to normalize the feature matrix

        learning_rate: float, between 0-1, default=0.01
            gradient descent step size
        '''
        if self.fit_called:
            raise ValueError('Fit method already called')

        self.fit_called = True
        # normalize
        if normalize:
            X = hp.normalize(X)

        # format np.arrays for regression
        X,y = hp.format_reg(X, y, normalized=normalize)

        if self.method == "least_squares":
            # fit through OLS
            coef = self.ols(X, y)

        else:
            # fit through gradient descent
            try:
                learning_rate <= 1
                learning_rate >= 0
            except:
                raise ValueError("Learning rate must be between 0-1")

            coef = self.gd(X, y)

        # convert to 1D array
        self.coef = coef.T.flatten()


    def get_coef(self, include_intercept=True):
        '''
        Return the fitted coefficients.

        Parameters
        ----------
        include_intercept: bool, default='True'
            whether to include the intercept coeff.
        '''
        if not self.fit_called:
            raise ValueError('Model not yet fit')

        if not include_intercept:
            return self.coef[1:]

        else:
            return self.coef
