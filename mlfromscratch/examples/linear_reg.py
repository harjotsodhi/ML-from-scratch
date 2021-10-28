from ..supervised.linear_regression import Linear_regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True)

# Use one feature for SLR and randomly sample some rows
np.random.seed(22)
ind = np.random.choice(X.shape[0], 30, replace=False)
X = X[ind,2].reshape(-1,1)
y = y[ind]

# Fit the custom LR model using OLS
model = Linear_regression(method="least_squares")
model.fit(X, y)

# Regression line visualized
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_plot = model.predict(X_plot)

# Plot outputs
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(X, y, color="black")
ax.plot(X_plot, y_plot, color="blue", label="Regression line")
ax.set_title("Simple linear regression model")
ax.set_xlabel("Body Mass Index (Standardized)")
ax.set_ylabel("Disease progression")
ax.legend()
ax.grid()
fig.savefig('mlfromscratch/examples/output/linear_reg.png')
plt.close(fig)
