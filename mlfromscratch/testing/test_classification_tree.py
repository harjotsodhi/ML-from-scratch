from ..supervised.cart import Classification_tree
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def test():
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # custom implementation
    clf = Classification_tree()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    # sklearn implementation
    sklearn_tree = tree.DecisionTreeClassifier(criterion='entropy',random_state=0)
    sklearn_tree.fit(X_train,y_train)
    y_pred_sklearn = sklearn_tree.predict(X_test)

    # test
    np.testing.assert_allclose(y_pred, y_pred_sklearn)


if __name__ == '__main__':
    test()
