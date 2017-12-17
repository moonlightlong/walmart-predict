#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

# model
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
from sklearn import tree

from sklearn import ensemble

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Feature Engineering
stores = pd.read_csv('stores.csv')


def v(x):
    d = {'A': 1, 'B': 2, 'C': 3}
    if x in d.keys():
        return d[x]
    else:
        return x
stores = stores.applymap(v)

train = pd.read_csv('train.csv')

features = pd.read_csv('features.csv')
features.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis=1, inplace=True)
features.fillna(method='ffill', inplace=True)

train_s = pd.merge(train, stores, how='left', on='Store')

train_s_f = pd.merge(train_s, features, how='left', on=['Store', 'Date', 'IsHoliday'])


def b2i(x):
    d = {False: 0, True: 1}
    if x in d.keys():
        return d[x]
    else:
        return x
train_s_f.IsHoliday = train_s_f.IsHoliday.map(b2i)

test = pd.read_csv('test.csv')
test_s = pd.merge(test, stores, how='left', on='Store')
test_s_f = pd.merge(test_s, features, how='left', on=['Store', 'Date', 'IsHoliday'])


def b2i(x):
    d = {False: 0, True: 1}
    if x in d.keys():
        return d[x]
    else:
        return x
test_s_f.IsHoliday = test_s_f.IsHoliday.map(b2i)

feature = ['IsHoliday', 'Type', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

mapper = DataFrameMapper([
    (['IsHoliday'], StandardScaler()),
    (['Type'], StandardScaler()),
    (['Size'], MinMaxScaler()),
    (['Temperature'], StandardScaler()),
    (['Fuel_Price'], StandardScaler()),
    (['CPI'], MinMaxScaler()),
    (['Unemployment'], StandardScaler())
])

with open('Submission.csv', 'w') as f:
    f.write('Id,Weekly_Sales\n')


def write(storeid, deptid, date, y):
    with open('Submission.csv', 'a') as f:
        for dateid, y_ in zip(list(date), y):
            line = str(storeid)+'_'+str(deptid)+'_'+str(dateid)+','+str(y_)+'\n'
            f.write(line)
            # print(line)

mission = pd.DataFrame(columns=['Id', 'Weekly_Sales'])
mae = list()
wmse = list()
ids = set()


# model
def model(train_x, train_y, test_X, flags='linear'):
    y = None
    if flags == 'linear':
        clf = linear_model.LinearRegression()

    elif flags == 'LSVR':
        clf = svm.LinearSVR()

    elif flags == 'SVR':
        clf = svm.SVR()

    elif flags == 'Ridge':
        clf = linear_model.Ridge()

    elif flags == 'TreeR':
        clf = tree.DecisionTreeRegressor()

    # Knn 不可以
    elif flags == 'Knn':
        clf = neighbors.KNeighborsRegressor()

    elif flags == 'RandomForest':
        clf = ensemble.RandomForestRegressor(n_estimators=20)

    elif flags == 'Adaboost':
        clf = ensemble.AdaBoostRegressor(n_estimators=50)

    elif flags == 'GBRT':
        clf = ensemble.GradientBoostingRegressor(n_estimators=100)

    else:
        pass
    clf.fit(train_x, train_y)
    y = clf.predict(test_X)
    return y
# model类型
flag = 'Adaboost'

for i in range(1, 46):
    dept_list = list(set(train_s_f[train_s_f.Store == i].Dept.values))
    test_dept_list = list(set(test_s_f[test_s_f.Store == i].Dept.values))
    # l = test_dept_list.copy()
    print('## sotre-{}: dept_num-{}'.format(i, len(dept_list)))
    for v in test_dept_list:
        if v not in dept_list:
            test_data = test_s_f[(test_s_f.Store == i) & (test_s_f.Dept == v)]
            date = test_data.Date

            train_data = train_s_f[train_s_f.Store == i]
            # X_data = train_data[feature].values
            # test_data = test_data[feature].values
            
            X_data = mapper.fit_transform(train_data[feature].applymap(lambda x: float(x)))
            test_data = mapper.fit_transform(test_data[feature].applymap(lambda x: float(x)))
            y_data = train_data['Weekly_Sales'].values

            y_predict = list(model(X_data, y_data, test_data, flags=flag))
            write(i, v, date, y_predict)
            test_dept_list.remove(v)
    dept_num = len(test_dept_list)
    print('   sotre-{}: dept_num-{}'.format(i, dept_num))
    if dept_num == 0:
        print(dept_list)
        print(l)
    for j in test_dept_list:
        train_data = train_s_f[(train_s_f.Store == i) & (train_s_f.Dept == j)]
        test_data = test_s_f[(test_s_f.Store == i) & (test_s_f.Dept == j)]

        # X_data = train_data[feature].values
        # test_data = test_data[feature].values
        X_data = mapper.fit_transform(train_data[feature].applymap(lambda x: float(x)))
        test_data = mapper.fit_transform(test_data[feature].applymap(lambda x: float(x)))
        y_data = train_data['Weekly_Sales'].values

        # 随机分配数据集以及model检测
        if len(X_data) == 0:
            print(j)
        if len(X_data) <= 3:
            X_train = X_data
            X_test = X_data[:1]
            y_train = y_data
            y_test = y_data[:1]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
        mae.append(mean_absolute_error(y_test, model(X_train, y_train, X_test, flags=flag)))

        y_predict = list(model(X_data, y_data, test_data, flags=flag))

        test_data = test_s_f[(test_s_f.Store == i) & (test_s_f.Dept == j)]
        date = test_data.Date
        write(i, j, date, y_predict)

print()
print('MAE: {}'.format(np.mean(np.asarray(mae))))
