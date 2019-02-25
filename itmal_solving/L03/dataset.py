#%% [markdown]
# ## Your Datasets
# 
# OK, now you need to find a dataset of your own! 
# 
# Goto www.kaggle.com and find a suitable dataset, that you want to work with. We want a set somewhat larger that iris or nmist but it should not be too big. 
# 
# You need to create an account at Kaggle before you download.
# 
# One example could be the beer consumption in Sao Paulo:
# 
# > https://www.kaggle.com/dongeorge/beer-consumption-sao-paulo/version/2
# 
# It is a 'small-data' set, download gives a 5Kb comma-separated file (CSV)...
# 
# 
# #### Qg Download a data set and do some data exploration of it
# 
# You are now a Data Scientist, go an examine your data, perhaps creating some feature scatter plots, just like the ones we just made for iris...
# 
# Are there `null`s or not-a-number data in your set? Do you have to filter these out before training?
# 
# Try to train-test split the set, perhaps just on a small set of its feature depending on the size of your data (small/medium/large/big), and try out one or two Scikit-learn ML algorithms on it just to see if it is possible.
# 
# (We return to the data set and training later...)

#%%
# TODO: Qg

import pandas as pd
import os

# Download and clean the data
print('Working directory: ' + os.getcwd())

coffee_data = pd.read_csv('Data/CoffeeAndCodeLT2018.csv')

print('Shape of raw CoffeeAndCodeLT2018.csv before preparing: ', coffee_data.shape)

cleaned_data = coffee_data.dropna()

print('Shape of Coffee Data after dropping columns with fields without values: ', cleaned_data.shape)

cleaned_data = coffee_data.drop("Country", axis=1) # Only one country
cleaned_data = cleaned_data.drop("AgeRange", axis=1) # Not gonna look at this
cleaned_data = cleaned_data.drop("CoffeeType", axis=1) # Not gonna look at this
cleaned_data = cleaned_data.drop("Gender", axis=1) # Not gonna look at this
cleaned_data = cleaned_data.drop("CodingWithoutCoffee", axis=1) # Not gonna look at this

print('Shape of Coffee Data after dropping columns we are not going to look at: ', cleaned_data.shape)

print('\nData sample: ')
cleaned_data.sample(5)

#
#%%
# Transform data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


encoder = LabelEncoder()

## Bugs solved by coffee encoding

coffee_bugs_cat = cleaned_data["CoffeeSolveBugs"]

coffee_bugs_encoded = encoder.fit_transform(coffee_bugs_cat)

print('Encoded classes:', encoder.classes_)
print('Encoded array:', coffee_bugs_encoded)

cleaned_data["CoffeeSolveBugs"] = coffee_bugs_encoded

cleaned_data.sample(1)
#

#%%
# One hot encoding for CoffeeTime
coffee_time_cat = cleaned_data["CoffeeTime"]

coffee_time_cat_encoded = encoder.fit_transform(coffee_time_cat)

print('Encoded classes:', encoder.classes_)
print('Encoded array:', coffee_time_cat_encoded)

hotencoder = OneHotEncoder()
coffee_time_cat_1hot = hotencoder.fit_transform(coffee_time_cat_encoded.reshape(-1,1))
print(hotencoder.categories_) # the classes are here renamed to number categories
print(coffee_time_cat_1hot.toarray())

prepared_coffee = cleaned_data.drop("CoffeeTime", axis=1) # Not gonna look at this
prepared_coffee

#

#%%
#import numpy as np
import matplotlib.pyplot as plt

prepared_coffee.plot(kind="scatter", x="CoffeeSolveBugs", y="CoffeeCupsPerDay")

#