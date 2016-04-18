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


def get_best(true, cur):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(true, cur)
    best = 0
    for i in range(len(precision)):
        if recall[i] >= 0.7:
            if precision[i] > best:
                best = precision[i]
    return best

data = pandas.read_csv("classification.csv", index_col=None)
true = data['true'].values
pred = data['pred'].values

tp = fp = tn = fn = 0

for i in range(len(true)):
    if true[i] == 1:
        if pred[i] == 1:
            tp += 1
        else:
            fn += 1
    else:
        if pred[i] == 0:
            tn += 1
        else:
            fp += 1

print tp, fp, fn, tn

print sklearn.metrics.accuracy_score(true, pred), sklearn.metrics.precision_score(true, pred), \
    sklearn.metrics.recall_score(true, pred), sklearn.metrics.f1_score(true, pred)

data = pandas.read_csv("scores.csv", index_col=None)
true = data['true'].values
score_logreg = data['score_logreg'].values
score_svm = data['score_svm'].values
score_knn = data['score_knn'].values
score_tree = data['score_tree'].values

print sklearn.metrics.roc_auc_score(true, score_logreg), \
    sklearn.metrics.roc_auc_score(true, score_svm), \
    sklearn.metrics.roc_auc_score(true, score_knn), \
    sklearn.metrics.roc_auc_score(true, score_tree)

print get_best(true, score_logreg), get_best(true, score_svm),\
    get_best(true, score_knn), get_best(true, score_tree)