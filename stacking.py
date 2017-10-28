import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import utils
import pandas as pd
import preprocess
import pickle
import os
from sklearn.metrics import confusion_matrix
import logging
logging.basicConfig(
    format='%(asctime)s :%(levelname)s : %(message)s',
    level=logging.INFO
)


seed = 520
# # Load the data set which have 210000
# print('Loading data ......')
# X = pd.read_csv('data/train.csv').iloc[:,1:]
# y = X['RACE']
# del X['RACE']
# del X['KEY']
# X = np.array(X)
# y = np.array(y).ravel()
# y = y - 1
# # load test
# X_test = pd.read_csv('data/ctest.csv').iloc[:,1:]
# y_test = X_test['RACE']
# del X_test['RACE']
# del X_test['KEY']
# X_test = np.array(X_test)
# y_test = np.array(y_test).ravel()
# y_test = y_test - 1
# X_train = X
# y_train = y
# index_data = np.isnan(np.array(pd.read_csv('data/test.csv')['RACE']))
# X_miss = X_test[index_data]
# y_miss = y_test[index_data]
# print('finish loading .....')

X_train, y_train, X_test, y_test, X_miss, y_miss, index_data = utils.get_data()

# # get the 5 fold to get the 5 models
indices = utils.get_kfold_indices(X_train)
if not os.path.exists('cache/rf.pkl'):
    if not os.path.exists('cache'):
        os.mkdir('cache')clf =
        logging.info('make cache dir ......')
    print('get models ......')
    # get the 5 fold to get the 5 models
    logging.info('Start splitting data set into 5 fold .....')
    models = []
    for i in range(5):
        logging.info('Grid search for the {} validation ......'.format(i))
        X_cvtrain, y_cvtrain, _, _ = utils.get_each_set(i, indices, X_train, y_train)
        params = utils.RF_para_search(X_cvtrain, y_cvtrain)
        RF_model1 = utils.train_RF(params, X_cvtrain, y_cvtrain)
        models.append(RF_model1)
    models_pkl = open('cache/rf.pkl', 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(models, models_pkl)
    models_pkl.close()
    logging.info('Finish saving the parameters')
else:
    logging.info('already did the grid search ....load parameters ....')
    fr = open('cache/rf.pkl', 'rb')
    models = pickle.load(fr)

logging.info('stacking it ......')
stacklayer1 = utils.stacking(models, indices, X_train)
stacklayer_test = utils.stacking_test(models,X_test)
stacklayer_miss = utils.stacking_test(models,X_miss)
print(stacklayer1.shape,stacklayer_test.shape,stacklayer_miss.shape)


import RF_baseline

logging.info('train without stacking ......')
clf_ = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf_.fit(X_train, y_train)
print("Without stacking Traing Score:%f" % clf_.score(X_train, y_train))
print("Testing Score:%f" % clf_.score(X_test, y_test))
print(confusion_matrix(y_test, clf_.predict(X_test)))
roc_auc, fpr, tpr = RF_baseline.compute_roc(y_test, clf_.predict(X_test), 6)
RF_baseline.save_plots(roc_auc, fpr, tpr, 6)

X_train = np.concatenate((X_train,stacklayer1),axis=1)
X_test = np.concatenate((X_test,stacklayer_test), axis=1)
X_miss = X_test[index_data]

logging.info('train with stacking, test different condition ......')
# X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.75, test_size=0.25, random_state=seed)
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf.fit(X_train, y_train)
print("Traing Score:%f" % clf.score(X_train, y_train))
print("Testing Score:%f" % clf.score(X_test, y_test))
print(confusion_matrix(y_test, clf.predict(X_test)))
roc_auc, fpr, tpr = RF_baseline.compute_roc(y_test, clf.predict(X_test), 6)
RF_baseline.save_plots(roc_auc, fpr, tpr, 6)


print("Missing Testing Score:%f" % clf.score(X_miss, y_miss))
print(confusion_matrix(y_miss, clf.predict(X_miss)))
roc_auc, fpr, tpr = RF_baseline.compute_roc(y_miss, clf.predict(X_miss), 6)
RF_baseline.save_plots(roc_auc, fpr, tpr, 6)


# train with stacking




# Only using stacklayer
# train stacklayer1  and y_train
from sklearn.neural_network import MLPClassifier
clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(6, 6), random_state=1)
clf2.fit(stacklayer1,y_train)
print("Traing Score:%f" % clf2.score(stacklayer1, y_train))
print("Testing Score:%f" % clf2.score(stacklayer_test, y_test))
print(confusion_matrix(y_test, clf2.predict(stacklayer_test)))
roc_auc, fpr, tpr = RF_baseline.compute_roc(y_test, clf2.predict(stacklayer_test), 6)
RF_baseline.save_plots(roc_auc, fpr, tpr, 6)



#
# logging.info('check single model best parameters: ......')
# params = utils.RF_para_search(X_train, y_train)
# RF_model1 = utils.train_RF(params, X_train, y_train)
# print("Traing Score:%f" % RF_model1.score(X_train, y_train))
# print("Testing Score:%f" % RF_model1.score(X_test, y_test))
# print(confusion_matrix(y_test, RF_model1.predict(X_test)))
# roc_auc, fpr, tpr = RF_baseline.compute_roc(y_test, RF_model1.predict(X_test), 6)
# RF_baseline.save_plots(roc_auc, fpr, tpr, 6)
