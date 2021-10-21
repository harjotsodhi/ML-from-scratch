import numpy as np
from ..solvers.gradient_descent import Gradient_descent
from ..helpers import helpers as hp



class Logistic_regression(object):
    """
    Logistic regression model.

    Parameters
    ----------
    normalized: bool
        whether features are normalized
        determines whether or not an intercept is needed.

    threshold: float, default=0.5
        probability threshold for classification

    learning_rate: float, default=0.01
        gradient descent step size (range 0-1)

    max_iter: int, default=1000
        maximum number of iterations for solver.

    abs_tol: float, default=1e-9
        absolute convergence tolerance for solver.
        end if: |cost_{n+1} - cost_{n}| < abs_tol
    """
    def __init__(self, method="gradient_descent", normalized=False,
                 threshold=0.5, learning_rate=0.01, max_iter=1000, abs_tol=1e-9):
        try:
            learning_rate <= 1
            learning_rate >= 0
        except:
            raise ValueError("Learning rate must be between 0-1")
        self.learning_rate = learning_rate
        self.threshold = threshold
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

        # fit through gradient descent
        coef = self._gd(X, y)

        # extract coefficients
        self.coef = coef


    def predict(self, X):
        '''
        Return the predicted value.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)
        '''
        # format np.arrays for regression
        X = hp.format_reg(X, normalized=self.normalized)
        self.pred_prob = 1/(1+np.exp(-(X.dot(self.coef))))
        y_pred = np.where(self.pred_prob > self.threshold, 1, 0)
        return y_pred.flatten()


    ### Private methods ###

    def _gd(self, X, y):
        '''
        Fit model using the gradient descent method.
        Encode cost and gradient as functions.

        prediction:
            => 1/1+exp{- X @ coef}
        cost:
            => (-y.T @ log{y_pred} - (1-y).T @ log{1 - y_pred})/m
        gradient:
            => ((y_pred-y).T @ X)/m

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        y: np.array
            response vector (m x 1)
        '''
        # prediction function
        def gd_predict(X, coef): return 1/(1+np.exp(-(X.dot(coef))))
        # MSE
        def gd_cost(m, y, y_pred): return (
                                            -y.T.dot(np.log(y_pred)) -
                                            (1-y).T.dot(np.log(1 - y_pred))
                                           ) * (1/m)
        # gradient
        def gd_gradient(m, X, y, y_pred): return ((y_pred-y).T.dot(X)).T * (1/m)
        # solve
        solver = Gradient_descent(gd_gradient, gd_cost, gd_predict,
                                  learning_rate=self.learning_rate,
                                  max_iter=self.max_iter, abs_tol=self.abs_tol)

        coef = solver.solve(X, y)
        return coef
