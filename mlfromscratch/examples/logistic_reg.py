from ..supervised.logistic_regression import Logistic_regression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from sklearn import datasets

# Load the iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Use just two features
X = X[:,:2]

# fit model
logit_reg = Logistic_regression(method="gradient_descent", normalized=False,
                                learning_rate=0.01, max_iter=10000, abs_tol=1e-9)

logit_reg.fit(X, y)

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
h = 0.01  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
mesh = np.c_[xx.ravel(), yy.ravel()]
Z = logit_reg.predict(mesh)

Z = Z.reshape(xx.shape)

# Plot outputs
fig, ax = plt.subplots(nrows=1, ncols=1)
colors = ['tab:blue','tab:orange','tab:green']
cMap = col.ListedColormap(colors)
ax.pcolormesh(xx, yy, Z, shading='auto', cmap=cMap)
cdict = {0:"Setosa", 1:"Versicolour", 2:"Virginica"}

for i in np.unique(y):
    ix = np.where(i == y)
    ax.scatter(X[ix, 0], X[ix, 1], label = cdict[i], edgecolors="k", color=colors[i])

ax.set_title("Multiclass logistic regression model")
ax.set_xlabel("Sepal length")
ax.set_ylabel("Sepal width")
ax.legend()
fig.savefig('mlfromscratch/examples/output/logistic_reg.png')
plt.close(fig)
