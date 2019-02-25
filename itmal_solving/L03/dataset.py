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

coffee_data = pd.read_csv(os.getcwd() + '\\Data\\CoffeeAndCodeLT2018.csv')

cleaned_data = coffee_data.drop("Country", axis=1) 
cleaned_data = cleaned_data.drop("AgeRange", axis=1) # Not gonna look at this
cleaned_data = cleaned_data.drop("CoffeeType", axis=1) # Not gonna look at this
cleaned_data = cleaned_data.drop("Gender", axis=1) # Not gonna look at this
cleaned_data = cleaned_data.drop("CodingHours", axis=1) # Don't know what they measuared

cleaned_data = cleaned_data.dropna()

cleaned_data.sample(1)

#
#%%
# Transform data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


encoder = LabelEncoder()

coffee_bugs_cat = cleaned_data["CoffeeSolveBugs"]

coffee_bugs_encoded = encoder.fit_transform(coffee_bugs_cat)

print(coffee_bugs_encoded)
print(encoder.classes_)

cleaned_data["CoffeeSolveBugs"] = coffee_bugs_encoded
cleaned_data.sample(1)
#

#%%
# One hot encoding for CoffeeTime
coffee_time_cat = cleaned_data["CoffeeTime"]

coffee_time_cat_encoded = encoder.fit_transform(coffee_time_cat)

print(coffee_time_cat_encoded)
print(encoder.classes_)

hotencoder = OneHotEncoder()
coffee_time_cat_1hot = hotencoder.fit_transform(coffee_time_cat_encoded.reshape(-1,1))
print(coffee_time_cat_1hot.toarray())
print(hotencoder.classes_)
#

#%%
#import numpy as np
import matplotlib.pyplot as plt

cleaned_data.plot(kind="scatter", x="CoffeeSolveBugs", y="CoffeeCupsPerDay")

#