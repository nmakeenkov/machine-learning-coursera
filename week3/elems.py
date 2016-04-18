import numpy as np
import pandas
import math
import sklearn
import sklearn.cross_validation
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.svm

data = pandas.read_csv("svm-data.csv", header=None)

X = data[[1, 2]]
y = data[0]

model = sklearn.svm.SVC(kernel='linear', C=100000, random_state=241)
model.fit(X, y)

print model.support_