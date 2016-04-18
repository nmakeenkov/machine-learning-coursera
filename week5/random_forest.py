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

data = pandas.read_csv("abalone.csv", index_col=None)

data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = data.values[:, :len(data.columns.values) - 1]
y = data['Rings'].values

fold = sklearn.cross_validation.KFold(len(X), n_folds=5, shuffle=True, random_state=1)

ans = -1
for n_est in range(1, 51):
    rgr = sklearn.ensemble.RandomForestRegressor(n_estimators=n_est, random_state=1)
    scores = sklearn.cross_validation.cross_val_score(rgr, X, y, cv=fold, scoring='r2')
    print scores.mean()
    if scores.mean() > .52 and ans == -1:
        ans = n_est

print ans
