# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 00:06:18 2017

@author: Mike
"""

import numpy as np 
import pandas as pd

train_df  = pd.read_json(open("train.json", "r"))
test_df = pd.read_json(open("test.json", "r"))

#print(train_df.tail())


# see the frequency of each feature
import collections
def most_common(lst):
    features = collections.Counter(lst)
    feature_value = features.keys()
    frequency = features.values()
    data = [('feature_value', feature_value),
            ('frequency', frequency),]    
    df = pd.DataFrame.from_items(data)
    return df.sort_values(by = 'frequency', ascending = False)


def newColumn(name,df,series):
    feature = pd.Series(0,df.index,name = name)# data : 0
    for row,word in enumerate(series):
        if name in word:
            feature.iloc[row] = 1
    df[name] = feature # feature : series ; value in series : 1 or 0
    return df

# select features based on frequency
facilities = ['Elevator','Cats Allowed','Hardwood Floors','Dogs Allowed','Doorman','Dishwasher','No Fee','Laundry in Building','Fitness Center']
for name in facilities:
    train_df = newColumn(name, train_df, train_df['features'])
print(train_df)



def newfeat(name, df, series):
    """Create a Series for my feature building loop to fill"""
    feature = pd.Series(0, df.index, name=name)
    """Now populate the new Series with numeric values"""
    for row, word in enumerate(series):
        if name in word:
            feature.iloc[row] = 1
    df[name] = feature
    return(df)
   
train_df = newfeat('Elevator', train_df, train_df.features)
train_df = newfeat('Dogs Allowed', train_df, train_df.features)
train_df = newfeat('Cats Allowed', train_df, train_df.features)
train_df = newfeat('Hardwood Floors', train_df, train_df.features)
train_df = newfeat('Dishwasher', train_df, train_df.features)
train_df = newfeat('Doorman', train_df, train_df.features)

train_df["created"] = pd.to_datetime(train_df["created"])
train_df["created_year"] = train_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
train_df["num_photos"] = train_df["photos"].apply(len)

train_df['price'] = train_df['price'].clip(upper=13000)
train_df["logprice"] = np.log(train_df["price"])


train_df["price_t"] =train_df["price"]/train_df["bedrooms"]


train_df["room_sum"] = train_df["bedrooms"]+train_df["bathrooms"] 


train_df['price_per_room'] = train_df['price']/train_df['room_sum']



train_df['latitude'] = round(train_df['latitude'], 2)
train_df['longitude'] = round(train_df['longitude'], 2)
train_df['latlong'] = train_df.latitude.map(str) + ', ' + train_df.longitude.map(str)
#print(len(train_df['latlong'].unique()))
test_df['latitude'] = round(test_df['latitude'], 2)
test_df['longitude'] = round(test_df['longitude'], 2)
test_df['latlong'] = test_df.latitude.map(str) + ', ' + test_df.longitude.map(str)

zipcode = pd.read_csv("zipcode.csv")


train_df= pd.merge(train_df, zipcode, how = 'left', on=['latlong'])
train_df = train_df.drop('void',1)
test_df = pd.merge(test_df, zipcode, how = 'left', on=['latlong'])
test_df = test_df.drop('void',1)


#print(train_zip.tail())

b_id = pd.concat([train_df['building_id'], test_df['building_id']]).unique()
b_id = pd.DataFrame(b_id)
b_id.columns = ['building_id']
b_id['building_index'] = [i for i in range(len(b_id))]
m_id = pd.concat([train_df['manager_id'], test_df['manager_id']]).unique()
m_id = pd.DataFrame(m_id)
m_id.columns = ['manager_id']
m_id['manager_index'] = [i for i in range(len(m_id))]
#print(m_id)
train_df= pd.merge(train_df, b_id, how = 'left', on=['building_id'])
train_df= pd.merge(train_df, m_id, how = 'left', on=['manager_id'])
test_df = pd.merge(test_df, b_id, how = 'left', on=['building_id'])
test_df = pd.merge(test_df, m_id, how = 'left', on=['manager_id'])
#print(train_zip.tail())












print(len(train_df))
#print(train_df.tail())

features_to_use = ["bathrooms", "bedrooms", "price",
             "num_photos", "Elevator", "Dogs Allowed",'Hardwood Floors','Cats Allowed'
             ,'Dishwasher','Doorman',
             "created_year", "created_month", "created_day",'latitude','longitude'
             ]

target_num_map = {'high':0, 'medium':1, 'low':2}
X = train_df[features_to_use]
y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
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


print(log_loss(y_val, y_val_pred))

print(accuracy_score(y_val, y_val_pred_acc))


'''
from sklearn import svm
rf2 = svm.SVC()
rf2.fit(X_train, y_train)
y_val_pred2 = rf2.predict_proba(X_val)
y_val_pred_acc2 = rf2.predict(X_val)




print(log_loss(y_val, y_val_pred2))

print(accuracy_score(y_val, y_val_pred_acc2))
'''



'''
train_df['pet_friendly'] = train_df['Cats Allowed'] + train_df['Dogs Allowed']
print(train_df['pet_friendly'])
'''


'''
feature_value = train_df['features'].tolist()

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