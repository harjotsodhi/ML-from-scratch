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
### Linear Regression

Linear regression is a supervised learning technique used for modeling a continuous
response variable as a linear function of one or more explanatory features. It can
be formalized mathematically as follows:

```math
y = X \beta
```

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
