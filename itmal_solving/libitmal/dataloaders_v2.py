#!/opt/anaconda3/bin/python

#import sys,os
#from itmal import utils

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata
from keras.datasets import mnist
from sklearn import datasets

def MOON_GetDataSet(n_samples=100, noise=0.1, random_state=0):
    X, y=datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y

def MOON_Plot(X, y):
    figure = plt.figure(figsize=(12, 9))
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k');

def MNIST_PlotDigit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")

def MNIST_GetDataSet():
    mnist = fetch_mldata('MNIST original')
    # NOTE: or (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    print("TODO: insert train/test split and shuffle code")
    #  ...
    #return ...    
    
# CEF: A production-code ready version of get MNIST
# Will try three different methods (sklearn v0.18, v0.20 and keras)
# 
def MNIST_GetDataSet(fetchmode=True, reshape784=True, debug=False):
    import numpy as np
    try:
        from sklearn.datasets import fetch_mldata
        has_datasets_fetch_mldata=True
    except:
        has_datasets_fetch_mldata=False    

    try:
        from sklearn.datasets import fetch_openml
        has_datasets_fetch_openml=True
    except:
        has_datasets_fetch_openml=False

    try:
        from keras.datasets import mnist
        has_kerasinstalled=True    
    except:
        has_kerasinstalled=False

    if fetchmode:
        if has_datasets_fetch_mldata:
            if debug:
                print("MNIST_GetDataSet(), in fetchmode, using fetch_mldata()...")
            # API Change Deprecated sklearn.datasets.fetch_mldata to be removed in 
            # version 0.22. mldata.org is no longer operational. Until removal it
            # will remain possible to load cached datasets. 
            #11466 by Joel Nothman.            
            d = fetch_mldata('MNIST original')
        elif has_datasets_fetch_openml:
            if debug:
                print("MNIST_GetDataSet(), in fetchmode, using fetch_openml()...")
            # scikit-learn v0.20.2
            d = fetch_openml('mnist_784', version=1, cache=True)
        else:
            raise ImportError("neither fetch_mldata() nor fetch_openml() was found in sklearn.datasets, so load of MNIST in fetchmod will not work!")
    
        X, y= d["data"], d["target"]          
    else:
        if debug:
            print("MNIST_GetDataSet(), in non-fetchmode, using keras mnist.load_data()...")            
        if has_kerasinstalled:            
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            X, y = np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test))
        else:
            raise ImportError("You do not have Keras installed, so keras.datasets.mnist.load_data() will not work!")
    
    if y.dtype!='uint8':
        y = y.astype('uint8')
        
    # NOTE: notice that X and y are defined inside if's, not in outer scope as in C++, strange!
    # NOTE: hardcoded sizes, 70000 x 28 x 28 or 70000 x 784
    assert X.ndim==2 or X.ndim==3
    assert (X.ndim==2 and X.shape[1]==784) or (X.ndim==3 and X.shape[1]==28 and X.shape[2]==28) 
    
    if reshape784 and X.ndim==2:
        assert X.shape[1]==784
        X=np.reshape(X, (70000, 28, 28))        
        assert X.ndim==3
        assert X.shape[1]==28 and X.shape[2]==28
   
    assert X.shape[0]==70000 
    assert X.shape[0]==y.shape[0]
    assert y.ndim==1
     
    assert X.dtype=='uint8'
    assert y.dtype=='uint8'
          
    return X, y

def Test_MNIST_GetDataSet():
    # Local function, pretty neat huh?
    def PrintShapeAndType(X, y, n):
        print('')
        print(f'X{n}.shape={X.shape}), X{n}.dtype={X.dtype}') 
        print(f'y{n}.shape={y.shape}), y{n}.dtype={y.dtype})') 
        
        assert X.shape[0]==y.shape[0]
        assert X.ndim==2 or X.ndim==3
        assert (X.ndim==2 and X.shape[1]==784) or (X.ndim==3 and X.shape[1]==28 and X.shape[2]==28) 
        
    X1, y1=MNIST_GetDataSet(fetchmode=True,  reshape784=True)
    X2, y2=MNIST_GetDataSet(fetchmode=False, reshape784=True)
    X3, y3=MNIST_GetDataSet(fetchmode=True,  reshape784=False)
    X4, y4=MNIST_GetDataSet(fetchmode=False, reshape784=False)
          
    PrintShapeAndType(X1, y1, '1')
    PrintShapeAndType(X2, y2, '2')
    PrintShapeAndType(X3, y3, '3')
    PrintShapeAndType(X4, y4, '4')
              
    # NOT test of ndim or X.shape[1], X1.shape[2]
    assert X1.shape[0]==X2.shape[0], f'unequal X shapes, X1.shape[0]={X1.shape[0]}, X2.shape[0]={X2.shape[0]}'
    assert X1.shape[0]==X3.shape[0], f'unequal X shapes, X1.shape[0]={X1.shape[0]}, X3.shape[0]={X3.shape[0]}'
    assert X1.shape[0]==X4.shape[0], f'unequal X shapes, X1.shape[0]={X1.shape[0]}, X4.shape[0]={X4.shape[0]}'
    assert y1.shape==y2.shape, f'unequal y shapes, y1.shape={y1.shape}, y2.shape={y2.shape}'
    assert y1.shape==y3.shape, f'unequal y shapes, y1.shape={y1.shape}, y3.shape={y3.shape}'
    assert y1.shape==y4.shape, f'unequal y shapes, y1.shape={y1.shape}, y4.shape={y4.shape}'
    assert type(X1)==type(X2), f'diff types, type(X1)={type(X1)}, type(X2)={type(X2)}' 
    assert type(X1)==type(X3), f'diff types, type(X1)={type(X1)}, type(X3)={type(X3)}' 
    assert type(X1)==type(X4), f'diff types, type(X1)={type(X1)}, type(X4)={type(X4)}' 
    assert type(y1)==type(y2), f'diff types, type(y1)={type(y1)}, type(y2)={type(y2)}' 
    assert type(y1)==type(y3), f'diff types, type(y1)={type(y1)}, type(y3)={type(y3)}' 
    assert type(y1)==type(y4), f'diff types, type(y1)={type(y1)}, type(y4)={type(y4)}' 
                                                                            
    assert X1.dtype==X2.dtype, f'diff dtypes, X1.dtype={X1.dtype}, X2.dtype={X2.dtype}' 
    assert X1.dtype==X3.dtype, f'diff dtypes, X1.dtype={X1.dtype}, X3.dtype={X3.dtype}' 
    assert X1.dtype==X4.dtype, f'diff dtypes, X1.dtype={X1.dtype}, X4.dtype={X4.dtype}' 
    assert y1.dtype==y2.dtype, f'diff dtypes, y1.dtype={y1.dtype}, y2.dtype={y2.dtype}' 
    assert y1.dtype==y3.dtype, f'diff dtypes, y1.dtype={y1.dtype}, y3.dtype={y3.dtype}' 
    assert y1.dtype==y4.dtype, f'diff dtypes, y1.dtype={y1.dtype}, y4.dtype={y4.dtype}'         
    
    #assert np.array_equal(X1,X3)
    assert np.array_equal(X2,X4)
    assert np.array_equal(y2,y4)    
    #assert (X1.ravel()==X2.ravel()).all()
    #MNIST_PlotDigit(X2[1])

def IRIS_GetDataSet():
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target
    return X, y 

def IRIS_PlotFeatures(X, y, i, j):
    plt.figure(figsize=(8, 6))
    plt.clf()
    plt.scatter(X[:, i], X[:, j], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.show()
         
def TrainTestSplit(X, y, N, shuffle=True, verbose=False):     
    assert X.shape[0]>N
    assert y.shape[0]==X.shape[0]
    assert X.ndim>=1
    assert y.ndim==1
    
    X_train, X_test, y_train, y_test = X[:N,:], X[N:,:], y[:N], y[N:] # or X[:N], X[N:], y[:N], y[N:]

    if shuffle:
        shuffle_index = np.random.permutation(N)
        X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    
    if verbose:
        print("X_train.shape=",X_train.shape)
        print("X_test .shape=",X_test.shape)
        print("y_train.shape=",y_train.shape)
        print("y_test .shape=",y_test.shape)
    
    return X_train, X_test, y_train, y_test

def Versions():    
    import sys
    print(f'{"Python version:":24s} {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}.')
    try:
        import sklearn as skl 
        print(f'{"Scikit-learn version:":24s} {skl.__version__}.')
    except:
        print(f'WARN: could not find sklearn!')  
    try:
        import keras as kr
        print(f'{"Keras version:":24s} {kr.__version__}')
    except:
        print(f'WARN: could not find keras!')  
    try:
        import tensorflow as tf
        print(f'{"Tensorflow version:":24s} {tf.__version__}')
    except:
        print(f'WARN: could not find tensorflow!')  
        
######################################################################################################
# 
# TESTS
#
######################################################################################################

def TestAll():
    Test_MNIST_GetDataSet()
    Versions()
    print("ALL OK")

if __name__ == '__main__':
    TestAll()
