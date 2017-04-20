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
#print(train_df.bathrooms.unique())
#print(test_df.bathrooms.unique())
#print(train_df.bedrooms.unique())
#print(test_df.bedrooms.unique())

#test_df["bathrooms"].loc[19671] = 1.5
#test_df["bathrooms"].loc[22977] = 2.0
#test_df["bathrooms"].loc[63719] = 2.0
#train_df["price"] = train_df["price"].clip(upper=13000)


#See the frequency of each feature and rank them based on frequency
print(train_df.features.unique())
print(len(train_df.features.unique()))
'''import collections
def most_common(lst):
    features = collections.Counter(lst)
    feature_value = features.keys()
    frequency = features.values()
    data = [('feature_value', feature_value),
            ('frequency', frequency),]    
    df = pd.DataFrame.from_items(data)
    return df.sort_values(by = 'frequency', ascending = False)'''


#Function to make a new column for features
def newColumn(name, df, series):
    feature = pd.Series(0, df.index, name = name)
    for row,word in enumerate(series):
        if name in word:
            feature.iloc[row] = 1
    df[name] = feature
    return df


#Select features based on frequency
facilities = ['Elevator','Cats Allowed','Hardwood Floors','Dogs Allowed','Doorman','Dishwasher','No Fee','Laundry in Building','Fitness Center'
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

#Create new attributes from price
train_df['price'] = train_df['price'].clip(upper=13000)
train_df["logprice"] = np.log(train_df["price"])
train_df["price_t"] =train_df["price"]/train_df["bedrooms"]
train_df["room_sum"] = train_df["bedrooms"]+train_df["bathrooms"]
train_df['price_per_room'] = train_df['price']/train_df['room_sum']


#Concatenate latitude and longitude into one column
train_df['latitude'] = round(train_df['latitude'], 2)
train_df['longitude'] = round(train_df['longitude'], 2)
train_df['latlong'] = train_df.latitude.map(str) + ', ' + train_df.longitude.map(str)
#print(len(train_df['latlong'].unique()))
test_df['latitude'] = round(test_df['latitude'], 2)
test_df['longitude'] = round(test_df['longitude'], 2)
test_df['latlong'] = test_df.latitude.map(str) + ', ' + test_df.longitude.map(str)

#Obtain zip code from unique latitude and longitude positions
'''l = pd.concat([train_df['latlong'], test_df['latlong']]).unique()
ll = pd.DataFrame(l)
#print(len(l))
l1.to_csv('C:/Users/tingt/PycharmProjects/BIA656/Final/neighborhood_new.csv')

from geopy.geocoders import Nominatim
geolocator = Nominatim()
#location = geolocator.reverse(train_df.iloc[484]['latlong'])
#location = geolocator.reverse(l[485])
#print(location.raw['address')

for i in range(581):
    location = geolocator.reverse(l[i])
    print(location.raw['address']['postcode'])
'''

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
features_to_use = ["bathrooms", "bedrooms", "price",
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
random_state = 5000
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.34, random_state = 5000)
rf1 = RandomForestClassifier(n_estimators=250, criterion='entropy',  n_jobs = 1,  random_state=random_state)
rf1.fit(X_train, y_train)
y_val_pred = rf1.predict_proba(X_val)
y_val_pred_acc = rf1.predict(X_val)
print(log_loss(y_val, y_val_pred))
print(accuracy_score(y_val, y_val_pred_acc))


#Logistic Regression
from sklearn.linear_model import LogisticRegression
rf2 = LogisticRegression()
rf2.fit(X_train, y_train)
y_val_pred2 = rf2.predict_proba(X_val)
y_val_pred_acc2 = rf2.predict(X_val)
print(log_loss(y_val, y_val_pred2))
print(accuracy_score(y_val, y_val_pred_acc2))

#Decision tree
from sklearn.tree import DecisionTreeClassifier
rf3 = DecisionTreeClassifier()
rf3.fit(X_train, y_train)
y_val_pred3 = rf3.predict_proba(X_val)
y_val_pred_acc3 = rf3.predict(X_val)
print(log_loss(y_val, y_val_pred3))
print(accuracy_score(y_val, y_val_pred_acc3))



#Naive Bayes
from sklearn.naive_bayes import GaussianNB
rf4 = GaussianNB()
rf4.fit(X_train, y_train)
y_val_pred4 = rf4.predict_proba(X_val)
y_val_pred_acc4 = rf4.predict(X_val)
print(log_loss(y_val, y_val_pred4))
print(accuracy_score(y_val, y_val_pred_acc4))

#Bagging
from sklearn.ensemble import BaggingClassifier
rf5 = BaggingClassifier()
rf5.fit(X_train, y_train)
y_val_pred5 = rf5.predict_proba(X_val)
y_val_pred_acc5 = rf5.predict(X_val)
print(log_loss(y_val, y_val_pred5))
print(accuracy_score(y_val, y_val_pred_acc5))

#KNN
from sklearn.neighbors import KNeighborsClassifier
rf6 =KNeighborsClassifier()
rf6.fit(X_train, y_train)
y_val_pred6 = rf6.predict_proba(X_val)
y_val_pred_acc6 = rf6.predict(X_val)
print(log_loss(y_val, y_val_pred6))
print(accuracy_score(y_val, y_val_pred_acc6))

#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
rf7 = AdaBoostClassifier(n_estimators=250)
rf7.fit(X_train, y_train)
y_val_pred6 = rf7.predict_proba(X_val)
y_val_pred_acc7 = rf7.predict(X_val)
print(log_loss(y_val, y_val_pred7))
print(accuracy_score(y_val, y_val_pred_acc7))


#Compare ROC of each Algorithm
import matplotlib.pyplot as plt
from sklearn import metrics
#RandomForest
fpr1, tpr1, threshold1 = metrics.roc_curve(y_val_pred_acc1, y_val_pred1)
roc_auc1 = metrics.auc(fpr1, tpr1)
plt.title('ROC of RandomForest')
plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.2f' % roc_auc1)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




#Using test dataset for submission
X_test = test_df[features_to_use]
y_test = rf1.predict_proba(X_test)
target_num_map = {'high':0, 'medium':1, 'low':2}
sub = pd.DataFrame()
sub["listing_id"] = test_df["listing_id"]
for label in ["high", "medium", "low"]:
    sub[label] = y_test[:, target_num_map[label]]
sub.to_csv("submission.csv", index=False)


'''#SVM
from sklearn.svm import SVC
rf2 = SVC()
rf2.fit(X_train, y_train)
y_val_pred2 = rf2.predict_proba(X_val)
y_val_pred_acc2 = rf2.predict(X_val)



print(log_loss(y_val, y_val_pred2))
print(accuracy_score(y_val, y_val_pred_acc2))




'''




