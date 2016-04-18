import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

sex = list(data['Sex'])
for i in range(len(sex)):
    if sex[i] == 'female':
        sex[i] = 0
    elif sex[i] == 'male':
        sex[i] = 1
    else:
        print sex[i], "KEKUS"

pclass = list(data['Pclass'])
fare = list(data['Fare'])
age = list(data['Age'])
survived = list(data['Survived'])

for i in range(len(sex)):
    if i >= len(sex):
        print i, " fin"
        break
    while i < len(sex) \
            and (np.isnan(pclass[i]) or np.isnan(fare[i]) or np.isnan(age[i]) or np.isnan(survived[i])):
        sex.pop(i)
        pclass.pop(i)
        fare.pop(i)
        age.pop(i)
        survived.pop(i)

tmp = []

for i in range(len(sex)):
    tmp.append([sex[i], pclass[i], fare[i], age[i]])

X = np.array(tmp)
y = np.array(survived)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

importances = clf.feature_importances_

print importances