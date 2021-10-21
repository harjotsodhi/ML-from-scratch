from ..supervised.logistic_regression import Logistic_regression
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def test():

    X, y = make_classification(n_samples=1000, n_features=5, n_informative=5,
                               n_redundant=0, n_classes=2, n_clusters_per_class=2,
                               flip_y=0.15, random_state=42)

    logit_reg = Logistic_regression(method="gradient_descent", normalized=False,
                                    learning_rate=0.01, max_iter=10000,
                                    abs_tol=1e-9)

    logit_reg.fit(X, y)

    sk_learn_clf = LogisticRegression(random_state=0, max_iter=1000, penalty="none")
    sk_learn_clf.fit(X, y)

    # check coefficients
    np.testing.assert_allclose(logit_reg.coef[1:].T, sk_learn_clf.coef_,
                               rtol=1e-03, atol=1e-03)


if __name__ == '__main__':
    test()
