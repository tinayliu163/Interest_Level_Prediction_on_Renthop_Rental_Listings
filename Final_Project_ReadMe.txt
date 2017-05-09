#use pandas to read json files
#read train and test dataset
train_df  = pd.read_json(open("train.json", "r"))
test_df = pd.read_json(open("test.json", "r"))


#Take out outliers for bedrooms, bathrooms, price from test dataset
test_df["bathrooms"].loc[19671] = 1.5
test_df["bathrooms"].loc[22977] = 2.0
test_df["bathrooms"].loc[63719] = 2.0


#Define two funtions to find out most common features in feature coloum
#See the frequency of each feature and rank them based on frequency
def most_common(lst):
    features = collections.Counter(lst)
    feature_value = features.keys()
    frequency = features.values()
    data = [('feature_value', feature_value),
            ('frequency', frequency),]    
    df = pd.DataFrame.from_items(data)
    return df.sort_values(by = 'frequency', ascending = False)

#Generate a new column for those features
def newColumn(name, df, series):
    feature = pd.Series(0, df.index, name = name)
    for row,word in enumerate(series):
        if name in word:
            feature.iloc[row] = 1
    df[name] = feature
    return df

#Based on the previous function to generate most frequent features in new coloumns
  with 1/0 as catogrical variable

facilities = ['Elevator','Cats Allowed','Hardwood Floors','Dogs Allowed','Doorman','Dishwasher','No Fee','Laundry in Building','Fitness Center',
             'Pre-War', 'Laundry in Unit', 'Roof Deck', 'Outdoor Space', 'Dining Room', 'High Speed Internet', 'Balcony', 'Swimming Pool']
for name in facilities:
    train_df = newColumn(name, train_df, train_df['features'])
    test_df = newColumn(name, test_df, test_df['features'])


#Convert date into numeric new variables
train_df["created"] = pd.to_datetime(train_df["created"])
train_df["created_year"] = train_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day

#Create new attributes from price
train_df['price'] = train_df['price'].clip(upper=13000)
train_df["logprice"] = np.log(train_df["price"])
train_df["price_t"] =train_df["price"]/train_df["bedrooms"]


#Create new attributes from room
train_df["room_sum"] = train_df["bedrooms"]+train_df["bathrooms"]
train_df['price_per_room'] = train_df['price']/train_df['room_sum']


#Using latittute and longtitute found out zip code of each apartment
then classify zip code and generate new columns
l = pd.concat([train_df['latlong'], test_df['latlong']]).unique()
ll = pd.DataFrame(l)
l1.to_csv('C:/Users/tingt/PycharmProjects/BIA656/Final/neighborhood_new.csv')
from geopy.geocoders import Nominatim
geolocator = Nominatim()
for i in range(581):
    location = geolocator.reverse(l[i])
    print(location.raw['address']['postcode'])
zipcode = pd.read_csv("neighborhood_new.csv")
#print(len(zipcode['postal_code'].unique()))
z_id = zipcode['postal_code'].unique()
z_id = pd.DataFrame(z_id)
z_id.columns = ['postal_code']
z_id['zip_id'] = [i for i in range(len(z_id))]
zipcode = pd.merge(zipcode, z_id, how = 'left', on = 'postal_code')
train_df= pd.merge(train_df, zipcode, how = 'left', on=['latlong'])
train_df = train_df.drop(['void', 'zip_code_index'], 1)


#Clustering buildings and managers to classify them into unique index
b_id = pd.concat([train_df['building_id'], test_df['building_id']]).unique()
b_id = pd.DataFrame(b_id)
b_id.columns = ['building_id']
b_id['building_index'] = [i for i in range(len(b_id))]
m_id = pd.concat([train_df['manager_id'], test_df['manager_id']]).unique()
m_id = pd.DataFrame(m_id)
m_id.columns = ['manager_id']
m_id['manager_index'] = [i for i in range(len(m_id))]
train_df= pd.merge(train_df, b_id, how = 'left', on=['building_id'])
train_df= pd.merge(train_df, m_id, how = 'left', on=['manager_id'])

#Define attributes and dependent variable
#Define X
features_to_use = ["bathrooms", "bedrooms", "price", 'logprice',"room_sum",
             "num_photos", "Elevator", "Dogs Allowed",'Hardwood Floors','Cats Allowed',
             'Dishwasher','Doorman', 'No Fee','Laundry in Building','Fitness Center',
             'Pre-War', 'Laundry in Unit', 'Roof Deck', 'Outdoor Space', 'Dining Room', 'High Speed Internet', 'Balcony', 'Swimming Pool',
             "created_year", "created_month", "created_day",'building_index', 'manager_index', 'zip_id'
             ]
X = train_df[features_to_use]

#Define Y
target_num_map = {'high':0, 'medium':1, 'low':2}
y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))


#Cross Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.34, random_state = 5000)

#Define different methods and classifiers that we wanted to use
#For example rf1 Random Forest
rf1 = RandomForestClassifier(n_estimators=250, criterion='entropy',  n_jobs = 1,  random_state=random_state)


#Use matplotlib to plot and compare different matrix of algorithms
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


#Confusion Matrix
confusion_matrix(y_val, y_val_pred_acc)


#Classification report
print(classification_report(y_val, y_val_pred_acc))

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


#Using test dataset for submission
X_test = test_df[features_to_use]
y_test = rf1.predict_proba(X_test)
sub = pd.DataFrame()
sub["listing_id"] = test_df["listing_id"]
for label in ["high", "medium", "low"]:
    sub[label] = y_test[:, target_num_map[label]]
sub.to_csv("submission.csv", index=False)
