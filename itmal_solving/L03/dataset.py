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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import Imputer

# Download data and view some of it
print('Working directory: ' + os.getcwd())

coffee_data = pd.read_csv(os.getcwd() + '\\Data\\CoffeeAndCodeLT2018.csv')

clean = coffee_data.drop("Country", axis=1) 

clean.sample(5)

#

#%%
