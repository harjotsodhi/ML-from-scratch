from ..supervised.linear_regression import Linear_regression
import numpy as np
from sklearn.linear_model import LinearRegression as sklearn_lm
from sklearn.datasets import make_regression


def test():

    X, y = make_regression(n_samples=100000, n_features=100, n_informative=75,
                           n_targets=1, noise=0.2, random_state=42)


    reg_sklearn = sklearn_lm()
    reg_sklearn.fit(X, y)

    # check OLS
    reg_ols = Linear_regression(method="least_squares")
    reg_ols.fit(X, y)

    # check coefficients
    np.testing.assert_allclose(reg_ols.get_coef()[1:], reg_sklearn.coef_,
                                rtol=1e-04, atol=1e-04)

    # check intercept
    np.testing.assert_allclose(reg_ols.get_coef()[0], reg_sklearn.intercept_,
                                rtol=1e-04, atol=1e-04)

    # check GD
    reg_GD = Linear_regression(method="gradient_descent")
    reg_GD.fit(X, y)

    # check coefficients
    np.testing.assert_allclose(reg_GD.get_coef(include_intercept=False), reg_sklearn.coef_,
                                rtol=1e-04, atol=1e-04)

    # check intercept
    np.testing.assert_allclose(reg_GD.get_coef(include_intercept=True)[0], reg_sklearn.intercept_,
                                rtol=1e-04, atol=1e-04)


if __name__ == '__main__':
    test()
