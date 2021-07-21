import numpy as np



class Gradient_descent(object):
    """
    Gradient descent algorithm to find optimal model parameters.
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
