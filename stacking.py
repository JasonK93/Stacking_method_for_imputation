import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import utils
import preprocess

seed = 520

# Load the data set which have 210000
_, _, X, y = preprocess.get_dummy()
X = np.array(X)[:10000]
y = np.array(y).ravel()[:10000]

# get the 5 fold
indices = utils.get_kfold_indices(X)

# test with the first stack
def get_each_set(i,indices):
    X_train = X[indices[i][0]]
    y_train = y[indices[i][0]]
    X_test = X[indices[i][1]]
    y_test = y[indices[i][1]]
    return X_train, y_train, X_test, y_test

models = []
X_train, y_train, _, _ = get_each_set(0,indices)
params = utils.RF_para_search(X_train, y_train)
RF_model1 = utils.train_RF(params, X_train, y_train)
models.append(RF_model1)







