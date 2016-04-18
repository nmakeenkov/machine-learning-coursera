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
import scipy.sparse

data_train = pandas.read_csv("salary-train.csv", index_col=None)
data_test = pandas.read_csv("salary-test-mini.csv", index_col=None)

data_train['FullDescription'] = data_train['FullDescription'].str.lower()
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
data_test['FullDescription'] = data_test['FullDescription'].str.lower()
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

enc = sklearn.feature_extraction.DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=5)
description_train = vectorizer.fit_transform(data_train['FullDescription'].values)
description_test = vectorizer.transform(data_test['FullDescription'].values)

X_train = scipy.sparse.hstack([description_train, X_train_categ])
X_test = scipy.sparse.hstack([description_test, X_test_categ])
y_train = data_train['SalaryNormalized'].values

model = sklearn.linear_model.Ridge(alpha=1, random_state=241)
model.fit(X_train, y_train)

print model.predict(X_test)