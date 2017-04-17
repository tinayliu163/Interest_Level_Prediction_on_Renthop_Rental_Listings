from geopy.geocoders import Nominatim
import pandas as pd
import csv
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances


#Load train data set
json_data=open('C:/Users/tingt/PycharmProjects/BIA656/Final/train.json').read()
train_df = pd.read_json(json_data)

#Load test data set
json_data_test=open('C:/Users/tingt/PycharmProjects/BIA656/Final/test.json').read()
test_df = pd.read_json(json_data_test)

#Convert latitude and longitude to zip codes
train_df['latitude'] = round(train_df['latitude'], 2)
train_df['longitude'] = round(train_df['longitude'], 2)
train_df['latlong'] = train_df.latitude.map(str) + ', ' + train_df.longitude.map(str)
#print(len(train_df['latlong'].unique()))
test_df['latitude'] = round(test_df['latitude'], 2)
test_df['longitude'] = round(test_df['longitude'], 2)
test_df['latlong'] = test_df.latitude.map(str) + ', ' + test_df.longitude.map(str)

'''l = train_df['latlong'].unique()
ll = pd.DataFrame(l)
print(len(l))
l1.to_csv('C:/Users/tingt/PycharmProjects/BIA656/Final/neighborhood.csv')
geolocator = Nominatim()
location = geolocator.reverse(train_df.iloc[0]['latlong'])
location = geolocator.reverse(l[153])
print(location.raw['address'])

for i in range(460):
    location = geolocator.reverse(l[i])
    print(location.raw['address']['postcode'])'''

zipcode = pd.read_csv("C:/Users/tingt/PycharmProjects/BIA656/Final/neighborhood.csv")
#print(zipcode.head())

#Join zipcode and train_df dataframes and create dummy variables
train_df= pd.merge(train_df, zipcode, how = 'left', on=['latlong'])
train_df = train_df.drop('void',1)
test_df = pd.merge(test_df, zipcode, how = 'left', on=['latlong'])
test_df = test_df.drop('void',1)
#print(test_df.postal_code)

#Count photos
train_df['number_photos'] = [len(train_df.iloc[i]['photos']) for i in range(len(train_df))]
test_df['number_photos'] = [len(test_df.iloc[i]['photos']) for i in range(len(test_df))]
#print(train_zip.tail())

#Create dummy variables for building and manager id
b_id = train_df['building_id'].unique()
b_id = pd.DataFrame(b_id)
b_id.columns = ['building_id']
b_id['building_index'] = [i for i in range(len(b_id))]
m_id = train_df['manager_id'].unique()
m_id = pd.DataFrame(m_id)
m_id.columns = ['manager_id']
m_id['manager_index'] = [i for i in range(len(m_id))]
#print(m_id)
train_df= pd.merge(train_df, b_id, how = 'left', on=['building_id'])
train_df= pd.merge(train_df, m_id, how = 'left', on=['manager_id'])
test_df = pd.merge(test_df, b_id, how = 'left', on=['building_id'])
test_df = pd.merge(test_df, m_id, how = 'left', on=['manager_id'])
#print(train_zip.tail())

#Features
feature_data = train_df[['description','features','interest_level']]
feature_value = feature_data['features'].tolist()
feature_lst = []
for i in range(len(feature_value)):
    feature_lst += feature_value[i]
# print(len(feature_lst)) # all features

uniq_feature = list(set(feature_lst))
# print(uniq_feature) #all unique features
len(uniq_feature)
# print(uniq_feature) #all unique features

# See the frequency of each feature
import collections
def most_common(lst):
    features = collections.Counter(lst)
    feature_value = features.keys()
    frequency = features.values()
    data = [('feature_value', feature_value),
            ('frequency', frequency),]
    df = pd.DataFrame.from_items(data)
    return df.sort_values(by = 'frequency', ascending = False)

#most_common(feature_lst)

#Create columns for most common features
def newColumn(name,df,series):
    feature = pd.Series(0,df.index,name = name)# data : 0
    for row,word in enumerate(series):
        if name in word:
            feature.iloc[row] = 1
    df[name] = feature # feature : series ; value in series : 1 or 0
    return df

# select features based on frequency
facilities = ['Elevator','Cats Allowed','Hardwood Floors','Dogs Allowed','Doorman','Dishwasher','No Fee','Laundry in Building','Fitness Center'
              'Pre-War', 'Laundry in Unit', 'Roof Deck', 'Outdoor Space', 'Dining Room', 'High Speed Internet', 'Balcony', 'Swimming Pool']
for name in facilities:
    train_df = newColumn(name, train_df, train_df['features'])
    test_df = newColumn(name, test_df, test_df['features'])
#print(train_df.head())
#print(test_df.head())


