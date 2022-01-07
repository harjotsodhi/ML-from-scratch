from mlfromscratch.supervised.random_forest import Random_forest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def test():
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # custom implementation
    rf = Random_forest()
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    accuracy = (y_test == y_pred).sum()/y_test.shape[0]

    # sklearn implementation
    rf_sklearn = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
    rf_sklearn.fit(X_train,y_train)
    y_pred_sklearn = rf_sklearn.predict(X_test)
    accuracy_sklearn = (y_test == y_pred_sklearn).sum()/y_test.shape[0]

    # test
    np.testing.assert_allclose(accuracy, accuracy_sklearn)


if __name__ == '__main__':
    test()
