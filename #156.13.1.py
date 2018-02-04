# #156.13.1.py

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
# fix random seed for reproducibility
np.random.seed(23)

from sklearn import datasets
(moon_x,moon_y) = datasets.make_moons(n_samples=10000, noise=0.05, random_state=42)
(circle_x,circle_y) = datasets.make_circles(n_samples=10000, noise=0.025, random_state=42)

# model = Sequential()
# model.add(Dense(12, input_dim=2, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))


# # moons_plot = np.asarray((moon_x,moon_y)).reshape(1,-1)

# # plt.plot((moon_x,moon_y))

# """
# For each dataset, plot the data in 2 dimensions, and also show the decision boundaries of your network 
# (for example, produce a visualization like: http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html).
# Now experiment with different neural network architectures and build classifiers for the two datasets given below. 
# For each dataset, the x is a 2 dimensional input and the y is the binary label 0 or 1.

# # Make note of your final accuracy on the training dataset, and the cross entropy score as well.
# """



def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)
    return out
xx, yy = make_meshgrid(moon_x[:,0], moon_x[:,1])
plot_contours(model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(moon_x[:,0], moon_x[:,1], c=moon_y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')


# ================================================================================================
###Ray
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
# fix random seed for reproducibility
np.random.seed(7)

# create model
model = Sequential()
model.add(Dense(12, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(moon_x, moon_y, epochs=10, batch_size=10)

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(1, 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0 = moon_x[:, 0]
X1 = moon_x[:, 1]
xx, yy = make_meshgrid(X0, X1)


plot_contours(plt, model, xx, yy,
                cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X0, X1, c=moon_y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

plt.show()
