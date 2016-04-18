import numpy as np
import pandas
import math
import sklearn
import sklearn.cross_validation
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.datasets

data = sklearn.datasets.load_boston()

X = data['data']
Y = data['target']

X = sklearn.preprocessing.scale(X)

fold = sklearn.cross_validation.KFold(len(X), n_folds=5, shuffle=True, random_state=42)

results = []

for p in np.linspace(1, 10, 200):
    model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance',
                                                  metric='minkowski', p=p)
    scores = sklearn.cross_validation.cross_val_score(model, X, Y, cv=fold,
                                                      scoring='mean_squared_error')
    results.append([scores.mean(), p])

ans = results[0]
for cur in results:
    if cur[0] > ans[0]:
        ans = cur

print ans