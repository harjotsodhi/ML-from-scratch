from ..supervised.linear_regression import Linear_regression
import numpy as np
from sklearn.linear_model import LinearRegression as sklearn_lm
from sklearn.datasets import make_regression

def _test():

    X, y = make_regression(n_samples=10000, n_features=100, n_informative=80,
                           n_targets=1, noise=0.2, random_state=42)


    reg_sklearn = sklearn_lm()
    reg_sklearn.fit(X, y)

    reg_custom = Linear_regression(method="least_squares")
    reg_custom.fit(X, y)

    # check coefficients
    np.testing.assert_allclose(reg_custom.get_coef()[1:], reg_sklearn.coef_,
                                rtol=1e-04, atol=1e-04)

    # check intercept
    np.testing.assert_allclose(reg_custom.get_coef()[0], reg_sklearn.intercept_,
                                rtol=1e-04, atol=1e-04)


if __name__ == '__main__':

    _test()
