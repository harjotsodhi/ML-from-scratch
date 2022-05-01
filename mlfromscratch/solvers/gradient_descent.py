import numpy as np
from ..helpers import helpers as hp



class Gradient_descent(object):
    """
    Gradient descent algorithm to find optimal model parameters.

    Parameters
    ----------
    gradient: function
        gradient encoded as a function

    cost: function
        cost encoded as a function

    predict: function
        prediction method encoded as a function

    learning_rate: float
        gradient descent step size (range 0-1)

    max_iter: int
        maximum number of iterations allowed

    abs_tol: float
        absolute convergence tolerance
            end if: |cost_{n+1} - cost_{n}| < abs_tol
    """
    def __init__(self, gradient, cost, predict,
                 learning_rate, max_iter, abs_tol):
        if [f for f in (gradient,cost,predict) if not callable(f)]: raise ValueError("Gradient, cost, and predict params must be functions")
        self.gradient_func,self.cost_func,self.predict_func = gradient,cost,predict
        if not 0 <= learning_rate <= 1: raise ValueError("Learning rate must be between 0-1")
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.abs_tol = abs_tol


    def solve(self, X, y):
        """
        Solve for optimal parameters.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        y: np.array
            response vector (m x 1)
        """
        m, n = X.shape
        # initialize coeffs at zero
        coef = np.zeros((n, 1))
        # initial prediction and cost
        y_pred = self.predict_func(X, coef)
        cost_prev = self.cost_func(m, y, y_pred)
        # initialize convergence criteria
        abs_diff = 1e99
        i = 0
        while (i < self.max_iter) and (abs_diff > self.abs_tol):
            # update coefficients
            coef = coef - self.learning_rate*self.gradient_func(m, X, y, y_pred)
            # new prediction and cost given updated coefficients
            y_pred = self.predict_func(X, coef)
            cost = self.cost_func(m , y, y_pred)
            # end early if change in cost is within convergence tolerance
            abs_diff = abs(cost_prev - cost)
            # iterate
            cost_prev = cost
            i += 1

        return coef
