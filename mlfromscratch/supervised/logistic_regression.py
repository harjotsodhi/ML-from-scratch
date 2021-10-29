import numpy as np
from ..solvers.gradient_descent import Gradient_descent
from ..helpers import helpers as hp



class Logistic_regression(object):
    """
    Logistic regression model.

    For multiclass problems, the one-vs-rest (ovr) approach is used.

    Parameters
    ----------
    normalized: bool
        whether features are normalized
        determines whether or not an intercept is needed.

    threshold: float, default=0.5
        probability threshold for binary classification problems.

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

        # apply a one-hot-encoder to the response variable
        encoded = hp.one_hot_encoder(y)
        # based on the output of one-hot-encoder, determine whether
        # the one-vs-rest (ovr) approach is needed
        self.coef = np.zeros((encoded.shape[1],X.shape[1]))
        for c in range(encoded.shape[1]):
            # solve each binary classification problem through gradient descent
            y_c = encoded[:,c].reshape(-1,1)
            coef_c = self._gd(X, y_c)
            self.coef[c,:] = coef_c.flatten()


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
        # calculate the predicted probability of membership to each class
        pred_prob = hp.sigmoid(X.dot(self.coef.T))
        # assign each X_i to the class with its highest predicting probability
        if pred_prob.shape[1] == 1:
            # binary case
            y_pred = np.where(pred_prob>self.threshold, 1, 0).flatten()
        else:
            # multi class case
            y_pred = np.argmax(pred_prob, axis=1)

        return y_pred


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
        def gd_predict(X, coef): return hp.sigmoid(X.dot(coef))
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
