import numpy as np
import pandas
import math
import sklearn
import sklearn.cross_validation
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.svm
import sklearn.datasets
import sklearn.feature_extraction.text
import sklearn.feature_extraction
import sklearn.decomposition
import scipy.sparse
import sklearn.ensemble
import matplotlib
import matplotlib.pyplot as plt

data = pandas.read_csv("gbm-data.csv", index_col=None)

X = data.values[:, 1:]
y = data.values[:, 0]

X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y,
                                                                             test_size=0.8,
                                                                             random_state=241)

for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
    clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=250, #verbose=True,
                                                      random_state=241, learning_rate=learning_rate)
    clf.fit(X_train, y_train)

    train_loss = []
    for y_pred in clf.staged_decision_function(X_train):
        train_loss.append(sklearn.metrics.log_loss(y_train, 1. / (1. + np.exp(-y_pred))))

    test_loss = []
    for y_pred in clf.staged_decision_function(X_test):
        test_loss.append(sklearn.metrics.log_loss(y_test, 1. / (1. + np.exp(-y_pred))))

    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.savefig("plot_" + str(learning_rate) + "_.png")

    if learning_rate == .2:
        best_iter = np.argmin(test_loss)
        print np.min(test_loss), np.argmin(test_loss)

clf = sklearn.ensemble.RandomForestClassifier(n_estimators=best_iter, random_state=241)
clf.fit(X_train, y_train)
print sklearn.metrics.log_loss(y_test, clf.predict_proba(X_test)[:, 1])