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


    def fit(self, X, y, normalized=False, learning_rate=0.01):
        '''
        Estimate the model parameters using the specified method.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        y: np.array
            response vector (m x 1)

        normalized: bool
            whether features are normalized
            determines whether or not an intercept is needed.

        learning_rate: float, default=0.01
            gradient descent step size (range 0-1)
        '''
        # format np.arrays for regression
        X,y = hp.format_reg(X, y, normalized=False)

        if self.method == "least_squares":
            # fit through OLS
            coef = self._ols(X, y)

        else:
            # fit through gradient descent
            try:
                learning_rate <= 1
                learning_rate >= 0
            except:
                raise ValueError("Learning rate must be between 0-1")

            coef = self._gd(X, y)

        # convert to 1D array
        self.coef = coef.T.flatten()


    def predict(self, X):
        '''
        Return the predicted value.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)
        '''
        y_pred = X.dot(self.coef)
        return y_pred


    ### Private methods ###

    def _ols(self, X, y):
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


    def _gd(self, X, y):
        '''
        Fit model using the gradient descent method.
        Encode cost and gradient as functions.

        _predict = X @ coef
        cost = MSE => SSE/2m => ((y_pred - y).T @ (y_pred - y))/2m
        gradient = partial derivative of MSE wrt coefs => (X.T @ (y_pred - y))/m

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        y: np.array
            response vector (m x 1)
        '''
        # prediction function
        def _predict(X, coef): return X.dot(coef)
        # MSE
        def cost(m, y, y_pred): return (1/(2*m)) * (y_pred-y).T.dot((y_pred-y))
        # gradient (partial derivative of MSE wrt coefs)
        def gradient(m, X, y, y_pred): return (1/m)*(X.T.dot((y_pred-y)))
        # solve
        solver = Gradient_descent(gradient, cost, _predict, max_iter=10000, abs_tol=1e-9)
        coef = solver.solve(X, y, learning_rate=0.01)
        return coef
