# -*- coding: utf-8 -*-
"""
Created on Wed March 15 16:35:38 2017

@author: Christina Eng, Ying Liu, Salman Sigari, Haoyue Yu
"""

import numpy as np
import pandas as pd


#Import train and test json files as dataframes
train_df  = pd.read_json(open("train.json", "r"))
test_df = pd.read_json(open("test.json", "r"))
#print(train_df.tail())

#Data Exploration
train_df.describe()
test_df.describe()

#Take out outliers for bedrooms, bathrooms, price
#print(train_df.bathrooms.value_counts())
#print(test_df.bathrooms.value_counts())
#print(train_df.bedrooms.value_counts())
#print(test_df.bedrooms.value_counts())
#print(train_df.price.value_counts().sort_index())
#print(test_df.price.value_counts().sort_index
#bath_out = [item for item in range(len(test_df['bathrooms'])) if test_df.iloc[item]['bathrooms'] >19]
#print(bath_out)
test_df["bathrooms"].loc[19671] = 1.5
test_df["bathrooms"].loc[22977] = 2.0
test_df["bathrooms"].loc[63719] = 2.0

#See the frequency of each feature and rank them based on frequency
import collections
def most_common(lst):
    features = collections.Counter(lst)
    feature_value = features.keys()
    frequency = features.values()
    data = [('feature_value', feature_value),
            ('frequency', frequency),]    
    df = pd.DataFrame.from_items(data)
    return df.sort_values(by = 'frequency', ascending = False)


#Function to make a new column for features
def newColumn(name, df, series):
    feature = pd.Series(0, df.index, name = name)
    for row,word in enumerate(series):
        if name in word:
            feature.iloc[row] = 1
    df[name] = feature
    return df


#Select features based on frequency
facilities = ['Elevator','Cats Allowed','Hardwood Floors','Dogs Allowed','Doorman','Dishwasher','No Fee','Laundry in Building','Fitness Center',
             'Pre-War', 'Laundry in Unit', 'Roof Deck', 'Outdoor Space', 'Dining Room', 'High Speed Internet', 'Balcony', 'Swimming Pool']
for name in facilities:
    train_df = newColumn(name, train_df, train_df['features'])
    test_df = newColumn(name, test_df, test_df['features'])
#print(train_df.head()


#Make attributes from created and photos column
train_df["created"] = pd.to_datetime(train_df["created"])
train_df["created_year"] = train_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
train_df["num_photos"] = train_df["photos"].apply(len)
#test_df
test_df["created"] = pd.to_datetime(test_df["created"])
test_df["created_year"] = test_df["created"].dt.year
test_df["created_month"] = test_df["created"].dt.month
test_df["created_day"] = test_df["created"].dt.day
test_df["num_photos"] = test_df["photos"].apply(len)

#Create new attributes from price
train_df['price'] = train_df['price'].clip(upper=13000)
train_df["logprice"] = np.log(train_df["price"])
train_df["price_t"] =train_df["price"]/train_df["bedrooms"]
train_df["room_sum"] = train_df["bedrooms"]+train_df["bathrooms"]
train_df['price_per_room'] = train_df['price']/train_df['room_sum']
#Test dataset
test_df['price'] = test_df['price'].clip(upper=13000)
test_df["logprice"] = np.log(test_df["price"])
test_df["price_t"] =test_df["price"]/test_df["bedrooms"]
test_df["room_sum"] = test_df["bedrooms"]+test_df["bathrooms"]
test_df['price_per_room'] = test_df['price']/test_df['room_sum']

#Concatenate latitude and longitude into one column
train_df['latitude'] = round(train_df['latitude'], 2)
train_df['longitude'] = round(train_df['longitude'], 2)
train_df['latlong'] = train_df.latitude.map(str) + ', ' + train_df.longitude.map(str)
#print(len(train_df['latlong'].unique()))
test_df['latitude'] = round(test_df['latitude'], 2)
test_df['longitude'] = round(test_df['longitude'], 2)
test_df['latlong'] = test_df.latitude.map(str) + ', ' + test_df.longitude.map(str)

#Obtain zip code from unique latitude and longitude positions
l = pd.concat([train_df['latlong'], test_df['latlong']]).unique()
ll = pd.DataFrame(l)
#print(len(l))
l1.to_csv('C:/Users/tingt/PycharmProjects/BIA656/Final/neighborhood_new.csv')
from geopy.geocoders import Nominatim
geolocator = Nominatim()
location = geolocator.reverse(train_df.iloc[484]['latlong'])
location = geolocator.reverse(l[485])
#print(location.raw['address')
for i in range(581):
    location = geolocator.reverse(l[i])
    print(location.raw['address']['postcode'])

#Import csv with zipcodes of unique latitude and longitude. Create id for unique zipcodes
zipcode = pd.read_csv("neighborhood_new.csv")
#print(len(zipcode['postal_code'].unique()))
z_id = zipcode['postal_code'].unique()
z_id = pd.DataFrame(z_id)
z_id.columns = ['postal_code']
z_id['zip_id'] = [i for i in range(len(z_id))]
zipcode = pd.merge(zipcode, z_id, how = 'left', on = 'postal_code')

#Merge zipcode and its id with train and test set
train_df= pd.merge(train_df, zipcode, how = 'left', on=['latlong'])
train_df = train_df.drop(['void', 'zip_code_index'], 1)
test_df = pd.merge(test_df, zipcode, how = 'left', on=['latlong'])
test_df = test_df.drop(['void', 'zip_code_index'], 1)
#print(train_df.head())

#Create index for unique building and manager ids, then merge with train and test set
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

#Define attributes and dependent variable
features_to_use = ["bathrooms", "bedrooms", "price", 'logprice',"room_sum",
             "num_photos", "Elevator", "Dogs Allowed",'Hardwood Floors','Cats Allowed',
             'Dishwasher','Doorman', 'No Fee','Laundry in Building','Fitness Center',
             'Pre-War', 'Laundry in Unit', 'Roof Deck', 'Outdoor Space', 'Dining Room', 'High Speed Internet', 'Balcony', 'Swimming Pool',
             "created_year", "created_month", "created_day",'building_index', 'manager_index', 'zip_id'
             ]

target_num_map = {'high':0, 'medium':1, 'low':2}
X = train_df[features_to_use]
y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

#Modeling
#Random Forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
random_state = 5000
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.34, random_state = 5000)
rf1 = RandomForestClassifier(n_estimators=250, criterion='entropy',  n_jobs = 1,  random_state=random_state)
rf1.fit(X_train, y_train)
y_val_pred = rf1.predict_proba(X_val)
y_val_pred_acc = rf1.predict(X_val)
logloss = log_loss(y_val, y_val_pred)
print(logloss)
accuracy = accuracy_score(y_val, y_val_pred_acc)
print(accuracy)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
rf2 = LogisticRegression()
rf2.fit(X_train, y_train)
y_val_pred2 = rf2.predict_proba(X_val)
y_val_pred_acc2 = rf2.predict(X_val)
logloss2 = log_loss(y_val, y_val_pred2)
print(logloss2)
accuracy2 = accuracy_score(y_val, y_val_pred_acc2)
print(accuracy2)

#Decision tree
from sklearn.tree import DecisionTreeClassifier
rf3 = DecisionTreeClassifier()
rf3.fit(X_train, y_train)
y_val_pred3 = rf3.predict_proba(X_val)
y_val_pred_acc3 = rf3.predict(X_val)
logloss3 = log_loss(y_val, y_val_pred3)
print(logloss3)
accuracy3 = accuracy_score(y_val, y_val_pred_acc3)
print(accuracy3)

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
rf4 = GaussianNB()
rf4.fit(X_train, y_train)
y_val_pred4 = rf4.predict_proba(X_val)
y_val_pred_acc4 = rf4.predict(X_val)
logloss4 = log_loss(y_val, y_val_pred4)
print(logloss4)
accuracy4 = accuracy_score(y_val, y_val_pred_acc4)
print(accuracy4)

#Bagging
from sklearn.ensemble import BaggingClassifier
rf5 = BaggingClassifier()
rf5.fit(X_train, y_train)
y_val_pred5 = rf5.predict_proba(X_val)
y_val_pred_acc5 = rf5.predict(X_val)
logloss5 = log_loss(y_val, y_val_pred5)
print(logloss5)
accuracy5 = accuracy_score(y_val, y_val_pred_acc5)
print(accuracy5)

#KNN
from sklearn.neighbors import KNeighborsClassifier
rf6 =KNeighborsClassifier()
rf6.fit(X_train, y_train)
y_val_pred6 = rf6.predict_proba(X_val)
y_val_pred_acc6 = rf6.predict(X_val)
logloss6 = log_loss(y_val, y_val_pred6)
print(logloss6)
accuracy6 = accuracy_score(y_val, y_val_pred_acc6)
print(accuracy6)

#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
rf7 = AdaBoostClassifier(n_estimators=250)
rf7.fit(X_train, y_train)
y_val_pred7 = rf7.predict_proba(X_val)
y_val_pred_acc7 = rf7.predict(X_val)
logloss7 = log_loss(y_val, y_val_pred7)
print(logloss7)
accuracy7 = accuracy_score(y_val, y_val_pred_acc7)
print(accuracy7)

#Compare different Algorithms
import matplotlib.pyplot as plt
accuracy_list = [accuracy, accuracy2, accuracy3, accuracy4, accuracy5
                 ,accuracy6,accuracy7]
accuracy_series = pd.Series.from_array(accuracy_list)
x_labels = ['RandomForest', 'LogisticRegression', 'Decision tree','Naive Bayes', 
            'Bagging', 'KNN','AdaBoost']
plt.figure(figsize=(8,5))
ax = accuracy_series.plot(kind='bar',color = '#ccccff')
ax.set_title("Accuracy Evaluation")
ax.set_xlabel("Methods")
ax.set_ylabel("Accuracy score")
ax.set_ylim([0.5,0.8])
ax.set_xticklabels(x_labels)
plt.show()

import matplotlib.pyplot as plt
logloss_list = [logloss, logloss2, logloss2, logloss3, logloss4
                 ,logloss6 ,logloss7]
logloss_series = pd.Series.from_array(logloss_list)
x_labels2 = ['RandomForest', 'LogisticRegression', 'Decision tree','Naive Bayes', 
            'Bagging', 'KNN','AdaBoost']
plt.figure(figsize=(8,5))
ax = logloss_series.plot(kind='bar',color = '#ffb3b3')
ax.set_title("Logloss Evaluation")
ax.set_xlabel("Methods")
ax.set_ylabel("Logloss")
ax.set_ylim([0,2])
ax.set_xticklabels(x_labels2)
plt.show()


from sklearn.metrics import confusion_matrix
#RF
confusion_matrix(y_val, y_val_pred_acc)

from sklearn.metrics import classification_report
#RF
print(classification_report(y_val, y_val_pred_acc))


#Using test dataset for submission
X_test = test_df[features_to_use]
y_test = rf1.predict_proba(X_test)
sub = pd.DataFrame()
sub["listing_id"] = test_df["listing_id"]
for label in ["high", "medium", "low"]:
    sub[label] = y_test[:, target_num_map[label]]
sub.to_csv("submission.csv", index=False)

#Feature Selection
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=29,
                              random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))
# Plot the feature importances of the forest
plt.title("Feature Importance", size = 30)
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], size = 16)
plt.xlim([-1, X.shape[1]])
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.show()

