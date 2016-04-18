import pandas
import math

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

_name = list(data['Name'])
sex = list(data['Sex'])

name = []

for i in range(len(sex)):
    if sex[i] == 'female':
        if 'Miss.' in _name[i] or 'Mrs.' in _name[i]:
            name.append(_name[i])

ans = pandas.DataFrame(columns=('Name', 'kek'))

for cur in name:
    cur_name = "KEK"
    if 'Miss.' in cur:
        cur_name = cur[cur.find('Miss') + len('Miss.') + 1:]
        if ' ' in cur_name:
            cur_name = cur_name[:cur_name.find(' ')]
    else:
        cur_name = cur[cur.find('Mrs') + len('Mrs.') + 1:]
        if '(' in cur_name:
            cur_name = cur_name[cur_name.find('(') + 1:]
        else:
            cur_name = "KEK"
        if ' ' in cur_name:
            cur_name = cur_name[:cur_name.find(' ')]
        if ')' in cur_name:
            cur_name = cur_name[:cur_name.find(')')]
    try:
        i = ans[cur_name]
    except KeyError:
        i = 0
    ans.loc[len(ans)] = [cur_name, 1]

print ans['Name'].value_counts()
