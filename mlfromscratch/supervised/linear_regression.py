import numpy as np
from mlfromscratch.solvers.gradient_descent import *
from mlfromscratch.helpers import helpers as hp



class Linear_regression(object):
    """
    Simple and multiple linear regression model.

    Parameters
    ----------
    method: str, default='gradient_descent'
        method for deriving parameters, 'least_squares' or 'gradient_descent'

    normalized: bool
        whether features are normalized
        determines whether or not an intercept is needed.

    learning_rate: float, default=0.01
        gradient descent step size (range 0-1)

    max_iter: int, default=1000
        maximum number of iterations for solver.

    abs_tol: float, default=1e-9
        absolute convergence tolerance for solver.
        end if: |cost_{n+1} - cost_{n}| < abs_tol
    """
    def __init__(self, method="gradient_descent", normalized=False,
                 learning_rate=0.01, max_iter=1000, abs_tol=1e-9):
        if method not in ("least_squares", "gradient_descent"):
            raise ValueError('Method param must be "least_squares" or "gradient_descent"')
        self.method = method
        try:
            learning_rate <= 1
            learning_rate >= 0
        except:
            raise ValueError("Learning rate must be between 0-1")
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.normalized = normalized


    def fit(self, X, y):
        '''
        Estimate the model parameters using the specified method.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        y: np.array
            response vector (m x 1)
        '''
        # format np.arrays for regression
        X,y = hp.format_reg(X, y, normalized=self.normalized)

        if self.method == "least_squares":
            # fit through OLS
            coef = self._ols(X, y)

        else:
            # fit through gradient descent
            coef = self._gd(X, y)

        # extract coefficients
        self.coef = coef


    def predict(self, X):
        '''
        Return the predicted value.

        y_pred = (w * X) + b

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)
        '''
        # format np.arrays for regression
        X = hp.format_reg(X, normalized=self.normalized)
        y_pred = X.dot(self.coef)
        return y_pred.flatten()


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

        prediction:
            => X @ coef
        cost = MSE:
            => SSE/m => ((y_pred - y).T @ (y_pred - y))/m
        gradient = partial derivative of MSE wrt coefs:
            => (X.T @ (y_pred - y))/m

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        y: np.array
            response vector (m x 1)
        '''
        # prediction function
        def gd_predict(X, coef): return X.dot(coef)
        # MSE
        def gd_cost(m, y, y_pred): return (1/m) * (y_pred-y).T.dot((y_pred-y))
        # gradient (partial derivative of MSE wrt coefs)
        def gd_gradient(m, X, y, y_pred): return (1/m)*(X.T.dot((y_pred-y)))
        # solve
        solver = Gradient_descent(gd_gradient, gd_cost, gd_predict,
                                  learning_rate=self.learning_rate,
                                  max_iter=self.max_iter, abs_tol=self.abs_tol)

        coef = solver.solve(X, y)
        return coef
