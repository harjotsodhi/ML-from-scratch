# Machine Learning From Scratch

## Project purpose
The purpose of this project is to implement a collection of machine learning
algorithms from scratch using only NumPy and the Python standard library.

This implementation focuses on writing code which is easy to understand and
highly accessible, as a learning exercise for readers to better understand the
inner workings of popular machine learning algorithms.

All of the algorithms presented in this repository feature extensively commented
code, reusable functions, theoretical explanations in this README, and have been
tested against the scikit-learn implementations.  

## Table of Contents
  - [Installation](#installation)
    * [With conda](#with-conda)
  - [Implementations](#implementations)
    * [Supervised Learning](#supervised-learning)
      + [Linear Regression](#linear-regression)
      + [Logistic Regression](#logistic-regression)
      + [CART](#cart)
      + [Random Forest](#random-forest)
    * [Unsupervised Learning](#unsupervised-learning)
      + [K-means](#k-means)
      + [PCA](#pca)
    * [Optimization](#optimization)
      + [Gradient descent](#Gradient-descent)
  - [Contact](#contact)

## Installation
### With conda
    $ git clone https://github.com/harjotsodhi/ML-from-scratch.git
    $ cd ML-from-scratch
    $ conda env create -f environment.yml
    $ conda activate ml_from_scratch_env

## Implementations
### Supervised Learning

#### Linear Regression

Linear regression is a supervised learning technique used for modeling a continuous
response variable as a linear function of one or more explanatory features and estimated
parameters. It can be formalized mathematically as follows:

![equation](https://latex.codecogs.com/gif.latex?y%20%3D%20Xw)

Where the vector Y (N by 1) contains a continuous response variable for each of the
N observations, the vector X (N by M+1) contains the M explanatory features plus
a column of ones representing the "bias" (i.e., the intercept), and the vector (M+1 by 1)
w contains the estimated parameters (i.e., the coefficients) of the linear regression model.

In linear regression, the model parameters can be estimated analytically through the method
of ordinary least squares, or through numerical methods such as Gradient descent.
This project's linear regression implementation includes options for solving both ways.

This project's linear regression implementation makes estimating the model parameters
and making predictions a piece of cake. An example implementation is provided below:

```python
from supervised.linear_regression import Linear_regression
# example of fitting the model through gradient descent
model = Linear_regression(method="gradient_descent", normalized=False,
                          learning_rate=0.01, max_iter=10000, abs_tol=1e-9)
model.fit(X, y)
y_pred = model.predict(Z)
```

The "mlfromscratch\\examples" subdirectory provides a variety of example implementations
with real data. The linear regression example shows the case of linear regression where
we are interested in predicting our response based off just one explanatory feature.
The resulting model can be seen visually below:

    $ python -m mlfromscratch.examples.linear_reg

<p align="center">
    <img src="https://github.com/harjotsodhi/ML-from-scratch/blob/master/mlfromscratch/examples/output/linear_reg.png?raw=true" width="640"\>
</p>
<p align="center">
    Figure 1: Linear regression model applied to a Diabetes dataset (https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf).
</p>

#### Logistic Regression

Logistic regression is a supervised learning technique used for modeling a categorical
response variable as a function of one or more explanatory features. Since, unlike in
linear regression, we are modeling a discrete rather than a continuous response variable,
the prediction task is known as "classification."

Logistic regression shares several key similarities with linear regression. In
logistic regression, we are interested in estimating the conditional probability
P(y=n | X_i) that observation X_i will be a member of class n. We do not assume
that the X_j explanatory features are a linear function of this probability. Instead,
we assume that the [logit](https://en.wikipedia.org/wiki/Logit) of this probability
is a linear function the explanatory features and estimated parameters.

Observations are predicted to belong to the class which has its highest predicted
probability. The implication of the logit of the probability
being a linear function the explanatory features and estimated parameters prediction task is
that we obtain a linear decision boundary.

Logistic regression can be generalized to classification tasks with more than
two classes. This project uses the One-vs.-rest (OvR) approach for handling
multiclass classification. This approach essentially breaks the problem into
k separate binary classification problems, where k is the number of classes
comprising the response variable.

In logistic regression, the model parameters must be estimated through numerical
methods such as Gradient descent.

This project's logistic regression implementation makes estimating the model parameters
and making predictions a piece of cake. An example implementation is provided below:

```python
from supervised.logistic_regression import Logistic_regression
# example of fitting the model through gradient descent
model = Logistic_regression(method="gradient_descent", normalized=False,
                            learning_rate=0.01, max_iter=1000, abs_tol=1e-9)
model.fit(X, y)
y_pred = model.predict(Z)
```

The "mlfromscratch\\examples" subdirectory provides a variety of example implementations
with real data. The logistic regression example shows the case of logistic regression where
we are interested in predicting a multiclass (k > 2) response based off two explanatory feature.
The resulting model can be seen visually below:

    $ python -m mlfromscratch.examples.logistic_reg

<p align="center">
    <img src="https://github.com/harjotsodhi/ML-from-scratch/blob/master/mlfromscratch/examples/output/logistic_reg.png?raw=true" width="640"\>
</p>
<p align="center">
    Figure 2: Logistic regression model applied to an Iris plants dataset (Fisher, R.A. “The use of multiple measurements in taxonomic problems” Annual Eugenics, 7, Part II, 179-188 (1936)).
</p>


#### CART

CART (classification and regression tree) is a supervised learning technique used
for modeling discrete response (in the case of classification trees) or continuous
response (in the case of regression tree) through decision trees.

CART models are conceptually simple. The features, X, are sequentially
split into two "branches" until the observations have been split into sufficiently
homogeneous groups (or a specified max depth is reached). CART models are considered
non-parametric learning algorithms, as the number of parameters in the model is not
pre-specified, and is rather based upon the data and will generally increase as
the size of the training data increases.

There are a variety of different splitting criteria for determining which feature
to split the dataset into next. In general, we want to use a splitting criteria
which maximizes the reduction in uncertainty, and is therefore "best." For classification
trees, [Entropy](https://en.wikipedia.org/wiki/Entropy#Information_theory) is commonly used,
and is used in this implementation.

CART models will often do very well at predicting training data (especially when
the model has no limitations on max number of splits) but are prone to overfitting.

In this implementation, CART models are built recursively. Using them is very easy,
an example implementation of a classification tree is provided below:

```python
from supervised.cart import Classification_tree
# example of fitting the classification tree
clf = Classification_tree(leaf_size=1)
clf.fit(X, y)
y_pred = clf.predict(Z)
```

#### Random Forest

Random Forest is an ensemble, supervised learning technique which builds upon
CART models. In the Random Forest algorithm, N decision trees are trained
upon M [bootstrapped](https://en.wikipedia.org/wiki/Bootstrap_aggregating) versions
of the original data set. For classification, the overall prediction is then the mode
of the N decision trees. For regression, the overall prediction is is the mean
prediction of the N decision trees.

Random Forest addresses the biggest shortfall of decision trees: overfitting.
Random Forest tends to perform better out-of-sample than decision trees because
each tree has been trained on a different version of the training data and thus
the variance of the overall model is reduced when averages are taken across the
N trees.

In this implementation, Random Forest models are built using N CART models
defined and implemented in the previous section. An example implementation of
a Random Forest built using 100 decision trees is provided below:

```python
from supervised.random_forest import Random_forest
# example of fitting the Random Forest
rf = Random_forest(num_trees=100)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
```

## Contact
Email: harjotsodhi17@gmail.com

LinkedIn: https://www.linkedin.com/in/harjot-sodhi/
