# Machine Learning From Scratch

## Project purpose
The purpose of this project is to implement a collection of machine learning
algorithms using only NumPy and the the Python standard library.

This implementation focuses on writing code which is easy to understand and
highly accessible, as a learning exercise for readers to better understand the
inner workings of popular machine learning algorithms.

All of the algorithms presented in this repository feature extensively commented
code, reusable functions, theoretical explanations in this README, and have been
tested against the scikit-learn implementations.  

## Table of Contents
  - [Installation](#installation)
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

## Installation (with conda)
    $ git clone https://github.com/harjotsodhi/ML-from-scratch.git
    $ cd ML-from-scratch
    $ conda env create -f environment.yml
    $ conda activate ml_from_scratch_env

## Implementations
### Supervised Learning

#### Linear Regression

Linear regression is a supervised learning technique used for modeling a continuous
response variable as a linear function of one or more explanatory features. It can
be formalized mathematically as follows:

![equation](https://latex.codecogs.com/gif.latex?y%20%3D%20Xw)

Where the vector (N by 1) Y is a contains a continuous response variable for the
N observations, the vector (N by M+1) X contains the M explanatory features plus
a column of ones representing the "bias" (i.e., the intercept), and the vector (M+1 by 1)
w contains the estimated parameters of the linear regression model.

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


## Contact
Email: harjotsodhi17@gmail.com

LinkedIn: https://www.linkedin.com/in/harjot-sodhi/
