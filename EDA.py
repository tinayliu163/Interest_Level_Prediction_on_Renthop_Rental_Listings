import numpy as np # linear algebra
import pandas as pd # data processing
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
### Seaborn style
sns.set_style("whitegrid")

house_data = pd.read_json(r'C:/Users/SONY/Desktop/BIA/StatisticalLearning/Group Project/two-sigma/train.json')
house_data.head()
print(house_data.head()
)
### Target variable exploration
sns.countplot(house_data.interest_level, order=['low', 'medium', 'high']);
plt.xlabel('Interest Level');
plt.ylabel('Number of occurrences');
          
'''### Number of unique Display Addresses
print('Number of Unique Display Addresses is {}'.format(house_data.display_address.value_counts().shape[0])) 
### 15 most popular Display Addresses
print(house_data.display_address.value_counts().nlargest(15))


### Rent interest graph of New-York
sns.lmplot(x="longitude", y="latitude", fit_reg=False, hue='interest_level',
           hue_order=['low', 'medium', 'high'], size=9, scatter_kws={'alpha':0.4,'s':30},
           data=house_data[(house_data.longitude>house_data.longitude.quantile(0.005))
                           &(house_data.longitude<house_data.longitude.quantile(0.995))
                           &(house_data.latitude>house_data.latitude.quantile(0.005))                           
                           &(house_data.latitude<house_data.latitude.quantile(0.995))]);
plt.xlabel('Longitude');
plt.ylabel('Latitude');

### Price exploration
fig = plt.figure(figsize=(12,12))
### Price distribution
sns.distplot(house_data.price[house_data.price<=house_data.price.quantile(0.99)], ax=plt.subplot(211));
plt.xlabel('Price');
plt.ylabel('Density');
### Average Price per Interest Level
sns.barplot(x="interest_level", y="price", order=['low', 'medium', 'high'],
            data=house_data, ax=plt.subplot(223));
plt.xlabel('Interest Level');
plt.ylabel('Price');
### Violinplot of price for every Interest Level
sns.violinplot(x="interest_level", y="price", order=['low', 'medium', 'high'],
               data=house_data[house_data.price<=house_data.price.quantile(0.99)],
               ax=plt.subplot(224));
plt.xlabel('Interest Level');
plt.ylabel('Price');      '''
