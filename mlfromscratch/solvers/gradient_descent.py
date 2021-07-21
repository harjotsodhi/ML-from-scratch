import numpy as np



class Gradient_descent(object):
    """
    Gradient descent algorithm to find optimal model parameters.

    Parameters
    ----------
    gradient: function
        gradient encoded as a function

    cost: function
        cost encoded as a function

    max_iter: int, default=10000
        maximum number of iterations allowed

    abs_tol: float, default=1e-9
        absolute convergence tolerance
            end if: |cost_{n+1} - cost_{n}| < abs_tol
    """
    def __init__(self, gradient, cost, max_iter=10000, abs_tol=1e-9):
        try:
            callable(gradient)
            callable(cost)
        except:
            raise ValueError("Gradient and cost params must be functions")

        self.gradient_func = gradient
        self.cost_func = cost
        self.max_iter = max_iter
        self.abs_tol = abs_tol


    def solve(self, X, y, learning_rate):
        """
        Solve for optimal parameters.

        Parameters
        ----------
        X: np.array
            feature matrix (m x n)

        y: np.array
            response vector (m x 1)

        learning_rate: float, between 0-1
            gradient descent step size
        """
        m, n = X.shape
        iter_ = 0
        # initialize coeffs at zero
        coef = np.zeros((n, 1))
        # initial prediction and cost
        y_pred = X.dot(coef)
        error = y_pred - y
        cost = self.cost_func(m , error)
        # initialize convergence criteria
        abs_diff = 1e99

        while (iter_ < self.max_iter) and (abs_diff > self.abs_tol):

            # update
            coef = coef - learning_rate*self.gradient_func(m, X, error)

            # new prediction and cost
            y_pred = X.dot(coef)
            error = y_pred - y
            cost_new = self.cost_func(m , error)

            # end early if change in cost is within convergence tolerance
            abs_diff = abs(cost - cost_new)

            # iterate
            cost = cost_new
            iter_ += 1

        return coef
