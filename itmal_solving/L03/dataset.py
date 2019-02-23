#%% [markdown]
# # ITMAL Exercise
# 
# REVISIONS|
# ---------|------------------------------------------------
# 2018-1219| CEF, initial.                  
# 2018-0206| CEF, updated and spell checked. 
# 2018-0206| CEF, added Kaggle dataset exercise. 
# 
# ## Vanilla Datasets
# 
# There are a number of popular datasets out-there, that are used again and again for small scale testing in ML: most popular are Moon, MNIST, Iris and CIFAR(10/100). We will use the three first here. 
# 
# (More on ML datasets: https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research)
# 
# ### Moon
# 
# <img src="Figs/moon.png" style="width:400px">
# 
# #### Qa Data load function 
# 
# We begin with a 100% synthetic dataset called moon. It creates two interleaved half-moon like datasets, and is frequently used as an XOR-like problem set (especially in Deep Learning).  
# 
# Create a `MOON_GetDataSet()` that generates moon data, based on Scikit-learn's `make_moon()` function.
# 
# Extend the `MOON_GetDataSet()`function signature to include some of the parameters found in `make_moon()`, like 'n_sample'.
# 
# Also create a `MOON_Plot()` function, that plots the data...good thing here is that the feature set is 2D and easy to handle!

#%%
# TODO: Qa...

# NOTE: some free help here regarding import clauses...
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

def MOON_GetDataSet(n_samples=100, shuffle=True, noise=None, random_state=None):
    # Simply use the make_moons function
    return make_moons(n_samples, shuffle, noise, random_state)

def MOON_Plot(X, y, title, xlabel, ylabel):
    # A scatter plot with first and second coloumns against y
    plt.scatter(X[:,0], X[:,1], s=40, c=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
               
# TEST CODE:
# Adds some noise to the data set
X, y = MOON_GetDataSet(n_samples=200, noise=0.05)
print("X.shape=",X.shape,", y.shape=",y.shape)

MOON_Plot(X,y, "Moon plot with noise", "x", "y")

#%% [markdown]
# #### Qb Try it with a  train-test split function
# 
# Now, use a train-test split function from Scikit-learn, that is able to split a `(X, y)` dataset tuple into a train-set and a test-set. 
# 
# Plot the train and test data using your `MOON_Plot` function.
# 
# Extend the plot function to add a plot title and x- and y-labels to the plot, say as default parameters
# 
# ```python 
# def MOON_Plot(X, y, title="my title", xlable="", ylabel=""):
#     # implementation here...
# ```
# 
# or similar. Use the titles "train" and "test" when plotting the train and test data.

#%%
# TODO: Qb....
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

MOON_Plot(X_train, y_train, "Moon plot train data", "x", "y")
plt.show()
MOON_Plot(X_test, y_test, "Moon plot test data", "x", "y")

#%% [markdown]
# ### MNIST
# 
# MNIST is a set of 70000 handwritten digits. It is used intensively as a form of "Hello World" dataset and estimator intensively check in Machine Learning. 
# 
# <img src="Figs/mnist.png" style="width:400px">
# 
# <!-- ![MNIST](https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/MnistExamples.png/220px-MnistExamples.png)-->
# 
# 
# https://en.wikipedia.org/wiki/MNIST_database
# 
# #### Qc Data load function 
# 
# Now for the MNIST data set, creating an easy to use data-loader, `MNIST_GetDataSet()`.
# 
# There are several ways of getting the MNIST dataset. You could base the data loader on the `fetch_mldata('MNIST original')` function or try to use the `keras.datasets.mnist.load_data()` function. 
# 
# The later function pre-splits into a train-test set, and to be compatible with the former, you must concatenate the train-test and return a plain `X, y` set. 
# 
# Also create a `MNIST_PlotDigit()`, that is able to plot a single digit from the dataset, and try to plot some of the digits in the dataset (set TEST CODE below).

#%%
from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_mldata
def GetMNISTRaw():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    print("Fetched raw MNIST")
    return mnist

mnist = GetMNISTRaw()


#%%
# TODO: Qc...
def MNIST_PlotDigit(data):
    image = data.reshape(28, 28)
    # TODO: add plot functionality for a single digit...
    plt.imshow(image, interpolation="nearest")
    plt.show()

def MNIST_GetDataSet(mnist):
    # TODO: use mnist = fetch_mldata('MNIST original') or mnist.load_data(),
    #       but return as a single X-y tuple 
    return mnist["data"], mnist["target"]

# TEST CODE:
X, y = MNIST_GetDataSet(mnist)
print("X.shape=",X.shape, ", y.shape=",y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=50000, shuffle=True)

print("X_train.shape=",X_train.shape,", X_test.shape=",X_test.shape)
MNIST_PlotDigit(X_train[5])
MNIST_PlotDigit(X_test[20])

#%% [markdown]
# ## Iris
# 
# Finally, for the iris data set: a four-dimension data set for 150 iris flowers, original created by  biologist Ronald Fisher and published via a paper in 1936.
# 
# <img src="Figs/iris.jpg" style="width:400px">
# 
# <!-- https://en.wikipedia.org/wiki/File:Iris_versicolor_3.jpg -->
# <!-- ![biologist Ronald Fisher](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/R._A._Fischer.jpg/200px-R._A._Fischer.jpg) -->
# 
# https://en.wikipedia.org/wiki/Iris_flower_data_set
# 
# #### Qd Data load function 
# 
# Creating the iris data loader, `IRIS_GetDataSet()`, this time we use the iris loader located in `sklearn.datasets.load_iris()`.

#%%
# TODO: Qd...
from sklearn.datasets import load_iris

def IRIS_GetDataSet():
    iris = load_iris()
    return iris["data"], iris["target"]

# TEST CODE:
X, y = IRIS_GetDataSet()

print("X.shape=",X.shape, ", y.shape=",y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, shuffle=True)

print("X_train.shape=",X_train.shape,", X_test.shape=",X_test.shape)

#%% [markdown]
# #### Qe Examine the data via scatter plots
# 
# Now, as a data-scientist, it is always good to get some grasp on how your data looks like. For the iris data we now want to plot some of the features-against-some-of-the other-features to see how they separate in the given 2D-feature space.
# 
# A scatter plot for all iris features against all other may look like
# 
# <img src="Figs/Iris_dataset_scatterplot.svg.png" style="width:400px">
# 
# Create a plot function that takes just two feature dimensions and plots them in a 2D plot, and plot all features against all others (resembling the "Iris Data" scatter plot just above).

#%%
# TODO: Qe...
def IRIS_PlotFeatureAgainstAll(X, y, featureNumber, featureLabel):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4,figsize=(20, 4))
 
    ax1.scatter(X[:,featureNumber], X[:,0], c=y)
    ax1.set_xlabel(featureLabel)
    ax1.set_ylabel("Sepal length$")
    
    ax2.scatter(X[:,featureNumber], X[:,1], c=y)
    ax2.set_xlabel(featureLabel)
    ax2.set_ylabel("Sepal width")
    
    ax3.scatter(X[:,featureNumber], X[:,2], c=y)
    ax3.set_xlabel(featureLabel)
    ax3.set_ylabel("Petal length")

    ax4.scatter(X[:,featureNumber], X[:,3], c=y)
    ax4.set_xlabel(featureLabel)
    ax4.set_ylabel("Petal width")

    plt.show()
    
def IRIS_PlotFeatures(X, y):
    IRIS_PlotFeatureAgainstAll(X, y, 0, "Sepal length")
    IRIS_PlotFeatureAgainstAll(X, y, 1, "Sepal width")
    IRIS_PlotFeatureAgainstAll(X, y, 2, "Petal length")
    IRIS_PlotFeatureAgainstAll(X, y, 3, "Petal width")
    
IRIS_PlotFeatures(X, y)

#%% [markdown]
# #### Qf Add your function to the `libitmal` python module
# 
# Add all your moon, MNIST and iris get and plot functions to the `libitmal` module. Call the file `dataloaders.py`, and test it in a ___new___ jupyter notebook (you need to reset the notebooks to be able to see if you are calling _cached_ version of the functions or the new ones, with similar names, in the lib module).
# 
# You will use these data loaders later, when we want to train on small datasets.

#%%
from libitmal import dataloaders
mnist = dataloaders.GetMNISTRaw()


#%%
# TODO: Qf...
X, y = dataloaders.MOON_GetDataSet(200, noise=0.05)

dataloaders.MOON_Plot(X, y, "moon plot", "x", "y")

X,y = dataloaders.MNIST_GetDataSet(mnist)

dataloaders.MNIST_PlotDigit(X[1000])

X, y = dataloaders.IRIS_GetDataSet()

dataloaders.IRIS_PlotFeatures(X, y)

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

get_ipython().run_line_magic('matplotlib', 'inline')

# Download data and view some of it
print('Working directory: ' + os.getcwd())

coffee_data = pd.read_csv(os.getcwd() + '\\Data\\CoffeeAndCodeLT2018.csv')

coffee_data.sample(5)

# 


#%%



