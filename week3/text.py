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

newsgroups = sklearn.datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
matrix = vectorizer.fit_transform(newsgroups.data)

X = matrix.toarray()

fold = sklearn.cross_validation.KFold(len(X), n_folds=5, shuffle=True,
                                      random_state=241)

best_c = 1
best_accuracy = -1
for c in 10 ** np.arange(-5., 6., 1.):
    model = sklearn.svm.SVC(kernel='linear', C=c, random_state=241)
    scores = sklearn.cross_validation.cross_val_score(model, matrix,
                                                      newsgroups.target, cv=fold)
    accuracy = scores.mean()
    if accuracy > best_accuracy:
        best_c = c
        best_accuracy = accuracy

model = sklearn.svm.SVC(kernel='linear', C=best_c, random_state=241)
model.fit(X, newsgroups.target)

x = model.coef_

values = []
for i in range(len(x[0])):
    values.append([abs(x[0][i]), i])

values.sort(key=(lambda x: -x[0]))

ans = []
for cur in values[:10]:
    ans.append(vectorizer.get_feature_names()[cur[1]])

ans.sort()

print ans