import numpy as np
import pandas
import math
import sklearn
import sklearn.cross_validation
import sklearn.neighbors
import sklearn.preprocessing

data = pandas.read_csv("wine.data", index_col=None)

X = data[['Alcohol', 'MalicAcid', 'Ash', 'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols', 'Flavanoids',
          'NonflavanoidPhenols', 'Proanthocyanins', 'ColorIntensity', 'Hue', 'OD280OD315', 'Proline']].values

X = sklearn.preprocessing.scale(X) # for task 3, 4

fold = sklearn.cross_validation.KFold(len(X), n_folds=5, shuffle=True, random_state=42)

results = []

for k in range(1, 51):
    model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    scores = sklearn.cross_validation.cross_val_score(model, X, data['Class'].values, cv=fold)
    results.append(scores.mean())

ans = 0
for i in range(50): # i = k - 1
    if results[i] > results[ans]:
        ans = i

print ans + 1, results[ans]