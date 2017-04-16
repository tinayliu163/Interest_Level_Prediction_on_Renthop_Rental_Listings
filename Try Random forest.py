# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 00:06:18 2017

@author: Mike
"""

import numpy as np 
import pandas as pd

sigma  = pd.read_json(open("train.json", "r"))

#print(sigma.tail())


def newfeat(name, df, series):
    """Create a Series for my feature building loop to fill"""
    feature = pd.Series(0, df.index, name=name)
    """Now populate the new Series with numeric values"""
    for row, word in enumerate(series):
        if name in word:
            feature.iloc[row] = 1
    df[name] = feature
    return(df)
   
sigma = newfeat('Elevator', sigma, sigma.features)
sigma = newfeat('Dogs Allowed', sigma, sigma.features)
sigma = newfeat('Cats Allowed', sigma, sigma.features)
sigma = newfeat('Hardwood Floors', sigma, sigma.features)
sigma = newfeat('Dishwasher', sigma, sigma.features)
sigma = newfeat('Doorman', sigma, sigma.features)

sigma["created"] = pd.to_datetime(sigma["created"])
sigma["created_year"] = sigma["created"].dt.year
sigma["created_month"] = sigma["created"].dt.month
sigma["created_day"] = sigma["created"].dt.day
sigma["num_photos"] = sigma["photos"].apply(len)
print(len(sigma))
#print(sigma.tail())


num_feats = ["bathrooms", "bedrooms", "price",
             "num_photos", "Elevator", "Dogs Allowed",'Hardwood Floors','Cats Allowed'
             ,'Dishwasher','Doorman',
             "created_year", "created_month", "created_day",'latitude','longitude']

target_num_map = {'high':0, 'medium':1, 'low':2}
X = sigma[num_feats]
y = np.array(sigma['interest_level'].apply(lambda x: target_num_map[x]))
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier 

random_state = 5000


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.34, random_state = 5000)


rf1 = RandomForestClassifier(n_estimators=250, criterion='entropy',  n_jobs = 1,  random_state=random_state)
rf1.fit(X_train, y_train)
y_val_pred = rf1.predict_proba(X_val)
y_val_pred_acc = rf1.predict(X_val)

from sklearn import svm
rf2 = svm.SVC()
rf2.fit(X_train, y_train)
y_val_pred2 = rf2.predict_proba(X_val)
y_val_pred_acc2 = rf2.predict(X_val)


print(log_loss(y_val, y_val_pred))

print(accuracy_score(y_val, y_val_pred_acc))

print(log_loss(y_val, y_val_pred2))

print(accuracy_score(y_val, y_val_pred_acc2))

'''
sigma['pet_friendly'] = sigma['Cats Allowed'] + sigma['Dogs Allowed']
print(sigma['pet_friendly'])
'''


'''
feature_value = sigma['features'].tolist()

feature_lst=[]
for i in range(len(feature_value)):
    feature_lst += feature_value[i]

mylist = list(feature_lst)
print(mylist)
print(len(mylist))

from collections import Counter
c=Counter(mylist)
def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(10)
print(Most_Common(mylist))
'''