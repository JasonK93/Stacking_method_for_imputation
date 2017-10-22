import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import utils
import preprocess
import logging
logging.basicConfig(
    format='%(asctime)s :%(levelname)s : %(message)s',
    level=logging.INFO
)


seed = 520
xx
# Load the data set which have 210000
_, _, X, y = preprocess.get_dummy()
X = np.array(X)  # [:10000]
y = np.array(y).ravel()  # [:10000]

# get the 5 fold
logging.info('Start splitting data set into 5 fold .....')
indices = utils.get_kfold_indices(X)

models = []
for i in range(5):
    logging.info('Grid search for the {} validation ......'.format(i))
    X_train, y_train, _, _ = utils.get_each_set(i, indices, X, y)
    params = utils.RF_para_search(X_train, y_train)
    RF_model1 = utils.train_RF(params, X_train, y_train)
    models.append(RF_model1)

logging.info('stacking it ......')
stacklayer1 = utils.stacking(models, indices, X, y)
print(stacklayer1.shape)


import RF_baseline
from sklearn.model_selection import train_test_split
X = stacklayer1
y = y
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.75, test_size=0.25, random_state=seed)
RF_baseline.test_RandomForestClassifier(X_train,X_test,y_train,y_test)






