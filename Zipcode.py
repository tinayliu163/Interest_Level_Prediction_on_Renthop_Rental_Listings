from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np



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
'''l = train_df['latlong'].unique()
print(len(l))
l.to_csv('C:/Users/tingt/PycharmProjects/BIA656/Final/neighborhood.csv')
geolocator = Nominatim()
location = geolocator.reverse(train_df.iloc[0]['latlong'])
location = geolocator.reverse(l[153])
print(location.raw['address'])

for i in range(460):
    location = geolocator.reverse(l[i])
    print(location.raw['address']['postcode'])'''

zipcode = pd.read_csv("C:/Users/tingt/PycharmProjects/BIA656/Final/neighborhood.csv")
#print(zipcode.head())

#Join dataframes
train_zip= pd.merge(train_df, zipcode, how = 'left', on=['latlong'])
train_zip = train_zip.drop('void',1)
train_zip = train_zip.drop('index',1)

#Count photos
train_zip['number_photos'] = [len(train_zip.iloc[i]['photos']) for i in range(49352)]





