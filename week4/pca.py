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

data = pandas.read_csv("close_prices.csv", index_col=None)

prices = data.values[:, 1:]

pca = sklearn.decomposition.PCA(n_components=0.9) # minimum variance value
prices_transformed = pca.fit_transform(prices)

print pca.n_components_

pca = sklearn.decomposition.PCA(n_components=10)
prices_transformed = pca.fit_transform(prices)

djia = pandas.read_csv("djia_index.csv", index_col=None)['^DJI'].values

print np.corrcoef(prices_transformed[:, 0], djia)

index = np.argmax(pca.components_[0])

print data.columns.values[index + 1]