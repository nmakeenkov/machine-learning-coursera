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


class GradientDescent:
    def __init__(self, w, k, x, y, l2_c=0):
        self.w = w
        self.k = k
        self.X = x
        self.y = y
        self.l2_c = l2_c

    def step(self):
        w = np.array([self.w[0], self.w[1]])
        plus = 0
        for i in range(len(self.X)):
            plus += y[i] * self.X[i][0] \
                    * (1.0 - 1.0 / (1.0 + math.exp(-self.y[i] * (self.w[0] * self.X[i][0] + self.w[1] * self.X[i][1]))))
        w[0] += plus * (self.k * 1.0 / len(self.X)) - self.k * self.l2_c * self.w[0]
        plus = 0
        for i in range(len(X)):
            plus += y[i] * X[i][1] \
                    * (1.0 - 1.0 / (1.0 + math.exp(-self.y[i] * (self.w[0] * self.X[i][0] + self.w[1] * self.X[i][1]))))
        w[1] += plus * (self.k * 1.0 / len(self.X)) - self.k * self.l2_c * self.w[1]
        self.w = w

    def solve(self):
        sqr = (lambda x : x * x)
        steps = 0
        diff = 1
        while steps < 10000 and diff > 1e-5:
            prev_w = np.array([self.w[0], self.w[1]])
            self.step()
            steps += 1
            diff = math.sqrt(sqr(self.w[0] - prev_w[0]) + sqr(self.w[1] - prev_w[1]))

    def probability(self, X):
        ans = []
        for x in X:
            ans.append(1. / (1. + math.exp(-self.w[0] * x[0] - self.w[1] * x[1])))
        return np.array(ans)


if __name__ == '__main__':
    data = pandas.read_csv("data-logistic.csv", header=None)

    X = data[[1, 2]].values
    y = data[0].values

    gd_no_l2 = GradientDescent(np.array([0.0, 0.0]), .1, X, y)
    gd_no_l2.solve()

    gd_l2 = GradientDescent(np.array([0.0, 0.0]), .1, X, y, l2_c=10)
    gd_l2.solve()

    print sklearn.metrics.roc_auc_score(y, gd_no_l2.probability(X)), ' ', \
        sklearn.metrics.roc_auc_score(y, gd_l2.probability(X))
