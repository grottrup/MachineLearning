import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.datasets import make_moons
from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_iris

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

def GetMNISTRaw():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    print("Fetched raw MNIST")
    return mnist

def MNIST_GetDataSet(mnist):
    # TODO: use mnist = fetch_mldata('MNIST original') or mnist.load_data(),
    #       but return as a single X-y tuple 
    return mnist["data"], mnist["target"].astype(int)

def MNIST_PlotDigit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, interpolation="nearest")
    plt.show()

def IRIS_GetDataSet():
    iris = load_iris()
    return iris["data"], iris["target"]

def IRIS_PlotFeatureAgainstAll(X, y, featureNumber, featureLabel):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4,figsize=(20, 4))

    ax1.scatter(X[:,featureNumber], X[:,0], c=y)
    ax1.set_xlabel(featureLabel)
    ax1.set_ylabel("Sepal length")
    
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