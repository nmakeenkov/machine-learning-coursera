import numpy as np
import pandas
import math
import sklearn
import sklearn.cross_validation
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.metrics

data_train = pandas.read_csv("perceptron-train.csv", header=None)
data_test = pandas.read_csv("perceptron-test.csv", header=None)

X_train = data_train[[1, 2]]
y_train = data_train[0]
X_test = data_test[[1, 2]]
y_test = data_test[0]

clf = sklearn.linear_model.Perceptron(random_state=241)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

accuracy_without_scaling = sklearn.metrics.accuracy_score(y_test, predictions)

scaler = sklearn.preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf_scaled = sklearn.linear_model.Perceptron(random_state=241)
clf_scaled.fit(X_train_scaled, y_train)
predictions = clf_scaled.predict(X_test_scaled)

accuracy_with_scaling = sklearn.metrics.accuracy_score(y_test, predictions)

print accuracy_with_scaling - accuracy_without_scaling
