# Week8.subset.svm.py

from sklearn import datasets, metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from time import time
import time
from sklearn.utils import shuffle
from numpy import arange
import random



def data_loader():
    digits = datasets.load_digits()
    print(digits.data.shape)
    return digits


def example_printing(data=data_loader):
    plt.gray()
    plt.matshow(data.images[0])
    plt.show()


def data_prep(digits=data_loader()):
    n_samples = len(digits.images)
    data_for_classifier = digits.images.reshape((n_samples, -1))
    return data_for_classifier, n_samples


def smaller_data_prep(digits=data_loader()):
    x_chosen = digits.images[np.where(np.logical_or(digits.target == 2, digits.target == 1))]
    y_chosen = digits.target[np.where(np.logical_or(digits.target == 2, digits.target == 1))]
    n_samples = len(x_chosen)
    data_for_classifier = x_chosen.reshape((n_samples, -1))
    return data_for_classifier, n_samples, y_chosen


def learn_small_data(kernel,data=smaller_data_prep()[0], n_samples=smaller_data_prep()[1], digits=smaller_data_prep()[0],
                     y=smaller_data_prep()[2]):
    classifier = SVC(probability=False,  cache_size=1000,  #TODO make sure that cache_size is not problematic
                     kernel=i, C=2.8, gamma='auto')
    classifier.fit(data[:n_samples // 2], y[:n_samples // 2])
    return classifier


def predict_small(classifier, n_samples=smaller_data_prep()[1]
                  , data=smaller_data_prep()[0], y=smaller_data_prep()[2]):
    expected = y[n_samples // 2:]
    predicted = classifier.predict(data[n_samples // 2:])
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


# SVM pipeline:
kernels = ['rbf', 'linear', 'poly']
for i in kernels:
    print(i)
    start_time = time.time()
    learn_small_data(data=smaller_data_prep()[0], n_samples=smaller_data_prep()[1],
                     digits=smaller_data_prep()[0],
                     y=smaller_data_prep()[2], kernel=i)
    print("running time is: '{0}' seconds --- for the kernel: '{1}".format({(time.time() - start_time)}, {i}))

    predict_small(classifier=learn_small_data(data=smaller_data_prep()[0], n_samples=smaller_data_prep()[1],
                                              digits=smaller_data_prep()[0],
                                              y=smaller_data_prep()[2], kernel=i), n_samples=smaller_data_prep()[1]
                  , data=smaller_data_prep()[0], y=smaller_data_prep()[2])
