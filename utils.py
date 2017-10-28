import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
import numpy as np
import pandas as pd
seed = 520

"""
* Get the total 6 data set we need
* Expect:
    None. Set the data path inside the function
* Return:
    -- A list: [X_train, y_train, X_test, y_test, X_miss, y_miss]
        - X train, y_train： train data. - X_test, y_test: total test data - X_miss, y_miss: the data having miss data in text.csv
    
"""
def get_data():
    # Load the data set which have 210000
    # print('Loading data ......')
    X = pd.read_csv('data/train.csv').iloc[:, 1:]
    y = X['RACE']
    del X['RACE']
    del X['KEY']
    X = np.array(X)
    y = np.array(y).ravel()
    y = y - 1
    # load test
    X_test = pd.read_csv('data/ctest.csv').iloc[:, 1:]
    y_test = X_test['RACE']
    del X_test['RACE']
    del X_test['KEY']
    X_test = np.array(X_test)
    y_test = np.array(y_test).ravel()
    y_test = y_test - 1
    X_train = X
    y_train = y
    index_data = np.isnan(np.array(pd.read_csv('data/test.csv')['RACE']))
    X_miss = X_test[index_data]
    y_miss = y_test[index_data]
    # print('finish loading .....')
    return [X_train, y_train, X_test, y_test, X_miss, y_miss, index_data]

"""
* Divide the data set into 5 part
* Expect:
    -- X = the data set 
* Return:
    -- indices = [[[train_index of first fold],[test_index of the first]],....],
                 [[first fold],[second fold],...]
"""
def get_kfold_indices(X):
    kf = KFold(n_splits=5, shuffle=True, random_state=520)
    indices = []
    for train, test in kf.split(X):
        indices.append([train, test])
    return indices


"""
* One function that use to find the best n_estimators
"""
def search_nestimator(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn import model_selection
    # RFC with fixed hyperparameters max_depth, max_features and min_samples_leaf
    rfc = RandomForestClassifier(n_jobs=-1, oob_score=False, max_depth=30, max_features='sqrt', min_samples_leaf=1)

    # Range of `n_estimators` values to explore.
    n_estim = [i for i in range(10, 100, 2)]
    cv_scores = []

    for i in n_estim:
        rfc.set_params(n_estimators=i)
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        scores = model_selection.cross_val_score(rfc, X_train, y_train, cv=kfold, scoring='accuracy')
        cv_scores.append(scores.mean() * 100)
    optimal_n_estim = n_estim[cv_scores.index(max(cv_scores))]
    print("The optimal number of estimators is %d with %0.1f%%" % (optimal_n_estim, max(cv_scores)))

    plt.plot(n_estim, cv_scores)
    plt.xlabel('Number of Estimators')
    plt.ylabel('Train Accuracy')
    plt.show()


def get_result(model, indices, X):
    pred = []
    for i in range(5):
        X_test = X[indices[i][1]]
        pred = pred + list(model.predict(X_test))
    return pred


"""
* Get the stack result from each models and combine the new train data
* Expects:
    -- models = all the models
    -- X_train, y_train = train data
* Return:
    -- stack_layer1 = 2D array, every column means one model learner
"""
def stacking(models,indices, X):
    results = []
    for model in models:
        pred = get_result(model, indices, X)
        results.append(pred)
    n = len(results)
    if n == 1:
        stack_layer1 = [[i] for i in results[0]]
    elif n >1:
        stack_layer1 = [[i] for i in results[0]]
        for i in range(1,n):
            tmp = [[j] for j in results[i]]
            stack_layer1 = np.concatenate((stack_layer1,tmp), axis= 1)
    else:
        raise MyException("No model found")
    return stack_layer1

def stacking_test(models,X):
    results = []
    for model in models:
        pred = list(model.predict(X))
        results.append(pred)
    n = len(results)
    if n == 1:
        stack_layer1 = [[i] for i in results[0]]
    elif n >1:
        stack_layer1 = [[i] for i in results[0]]
        for i in range(1,n):
            tmp = [[j] for j in results[i]]
            stack_layer1 = np.concatenate((stack_layer1,tmp), axis= 1)
    else:
        raise MyException("No model found")
    return stack_layer1



def RF_para_search(X_train, y_train):
    rfc = RandomForestClassifier(n_jobs=-1, oob_score=False, max_depth=30, max_features='sqrt', min_samples_leaf=1)
    # Use a grid over parameters of interest
    param_grid = {
        "n_estimators": [36, 45, 54, 63, 72],
        "max_depth": [5, 10, 15, 20, 25, 30, 35, 40],
        "min_samples_leaf": [1, 2, 4, 6, 8, 10]}

    # param_grid = {
    #     "n_estimators": [50],
    #     "max_depth": [30],
    #     "min_samples_leaf": [1]}

    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10)
    CV_rfc.fit(X_train, y_train)
    params = CV_rfc.best_params_
    print('best para for this part:', params)
    print('best score:', CV_rfc.best_score_)
    return params


def train_RF(params, X_train, y_train):
    rfc = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                 max_features='sqrt', min_samples_leaf=params['min_samples_leaf'])

    rfc.fit(X_train,y_train)

    return rfc


def get_each_set(i, indices, X, y):
    X_train = X[indices[i][0]]
    y_train = y[indices[i][0]]
    X_test = X[indices[i][1]]
    y_test = y[indices[i][1]]
    return X_train, y_train, X_test, y_test


"""
*
*
*
"""
# Todo: ADD more base estimator
def Ada_para_search(X_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    ada_dtc = DecisionTreeClassifier()
    ada = AdaBoostClassifier(base_estimator = ada_dtc)

    param_grid = {"learning_rate": [1,0.1,0.01],
                  "n_estimators": [25, 50]}
    CV_ada = GridSearchCV(n_jobs=-1, estimator=ada,param_grid=param_grid,cv=10)
    CV_ada.fit(X_train,y_train)
    params = CV_ada.best_params_
    print('best para for this part:', params)
    print('best score:', CV_ada.best_score_)
    return params

def train_ada(params, X_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    ada_dtc = DecisionTreeClassifier()
    ada = AdaBoostClassifier(base_estimator=ada_dtc,learning_rate=params['learning_rate'], n_estimators=params['n_estimators'])
    ada.fit(X_train,y_train)
    return ada



# Elastic
def Elastic_para_search(X_train, y_train):
    from sklearn.linear_model import ElasticNet
    elastic =ElasticNet(alpha=1.0,l1_ratio=0.5)

    param_grid = {"alpha": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                  "l1_ratio": [0.3, 0.5, 0.8]}
    CV_elastic = GridSearchCV(n_jobs=-1, estimator=elastic,param_grid=param_grid,cv=10)
    CV_elastic.fit(X_train,y_train)
    params = CV_elastic.best_params_
    print('best para for this part:', params)
    print('best score:', CV_elastic.best_score_)
    return params


def train_elastic(params, X_train, y_train):
    from sklearn.linear_model import ElasticNet
    elastic = ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'])
    elastic.fit(X_train,y_train)
    return elastic


# SGD
def sgd_para_search(X_train, y_train):
    from sklearn.linear_model import SGDClassifier
    sgd = SGDClassifier(n_jobs=-1, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15,
                        n_iter=5, eta0=0.0, power_t=0.5)

    # param_grid = {"alpha": [0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.01],
    #               "l1_ratio": [0.13, 0.15, 0.18],
    #               "n_iter": [5, 100, 200, 500],
    #               "penalty":['l1', 'l2', 'elasticnet']}
    #
    param_grid = {"alpha": [0.0001],
                  "l1_ratio": [0.13],
                  "n_iter": [500],
                  "penalty":['elasticnet']}

    CV_sgd = GridSearchCV(n_jobs=-1, estimator=sgd, param_grid=param_grid, cv=10)
    CV_sgd.fit(X_train,y_train)
    params = CV_sgd.best_params_
    print('best para for this part:', params)
    print('best score:', CV_sgd.best_score_)
    return params


def train_sgd(params, X_train, y_train):
    from sklearn.linear_model import SGDClassifier
    sgd = SGDClassifier(n_jobs=-1, loss='hinge', penalty=params['penalty'], alpha=params["alpha"], l1_ratio=params["l1_ratio"],
                        n_iter=params["n_iter"], eta0=0.0, power_t=0.5)

    sgd.fit(X_train,y_train)
    return sgd


# KNN
def knn_para_search(X_train, y_train):
    from sklearn import  neighbors
    knn = neighbors.KNeighborsClassifier(neighbors= 6,weights='uniform', algorithm='auto',
                                         leaf_size=30, p =2, metric='minkowski', n_jobs=-1)


    param_grid = {"leaf_size": [10,20,30,40,50],
                  "p": [1, 2, 3, 4],
                  "weights": ['uniform', 'distance'],
                  "algorithm":['auto', 'ball_tree', 'kd_tree', 'brute']}
    CV_knn = GridSearchCV(n_jobs=-1, estimator=knn, param_grid=param_grid, cv=10)
    CV_knn.fit(X_train,y_train)
    params = CV_knn.best_params_
    print('best para for this part:', params)
    print('best score:', CV_knn.best_score_)
    return params


def train_knn(params, X_train, y_train):
    from sklearn import  neighbors
    knn = neighbors.KNeighborsClassifier(neighbors= 6,weights=params['weights'], algorithm=params['algorithm'],
                                         leaf_size=params['leaf_size'], p =params['p'], metric='minkowski', n_jobs=-1)

    knn.fit(X_train,y_train)
    return knn


# GPC
def gpc_para_search(X_train, y_train):
    from sklearn.gaussian_process import GaussianProcessClassifier

    gpc = GaussianProcessClassifier(kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0,max_iter_predict=100,n_jobs=-1)

    param_grid = {"n_restarts_optimizer": [0, 5, 10],
                  "max_iter_predict": [100, 200, 500]}
    CV_gpc = GridSearchCV(n_jobs=-1, estimator=gpc, param_grid=param_grid, cv=10)
    CV_gpc.fit(X_train,y_train)
    params = CV_gpc.best_params_
    print('best para for this part:', params)
    print('best score:', CV_gpc.best_score_)
    return params


def train_gpc(params, X_train, y_train):
    from sklearn.gaussian_process import GaussianProcessClassifier

    gpc = GaussianProcessClassifier(kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=params['n_restarts_optimizer'],max_iter_predict=params['max_iter_predict'],n_jobs=-1)

    gpc.fit(X_train,y_train)
    return gpc


# MNB
def mnb_para_search(X_train, y_train):
    from sklearn.naive_bayes import MultinomialNB
    mnb = MultinomialNB(alpha=1.0,class_prior=[6,5,4,3,2,1])

    param_grid = {"alpha": [0, 0.5, 1.0]}
    CV_mnb = GridSearchCV(n_jobs=-1, estimator=mnb, param_grid=param_grid, cv=10)
    CV_mnb.fit(X_train, y_train)
    params = CV_mnb.best_params_
    print('best para for this part:', params)
    print('best score:', CV_mnb.best_score_)
    return params


def train_mnb(params, X_train, y_train):
    from sklearn.naive_bayes import MultinomialNB
    mnb = MultinomialNB(alpha=params['alpha'],class_prior=[6,5,4,3,2,1])

    mnb.fit(X_train,y_train)
    return mnb


# MLP
def mlp_para_search(X_train, y_train):
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(100,100,50,),activation='relu', solver='adam',
                        alpha=0.0001, batch_size='auto', learning_rate="constant", learning_rate_init=0.001,
                        max_iter=5000, momentum=0.9, validation_fraction=0.1,shuffle=True)

    param_grid = {"learning_rate_init": [0.0001, 0.001, 0.01, 0.1, 1]}
    CV_mlp = GridSearchCV(n_jobs=-1, estimator=mlp, param_grid=param_grid, cv=10)
    CV_mlp.fit(X_train, y_train)
    params = CV_mlp.best_params_
    print('best para for this part:', params)
    print('best score:', CV_mlp.best_score_)
    return params


def train_mlp(params, X_train, y_train):
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 50,), activation='relu', solver='adam',
                        alpha=0.0001, batch_size='auto', learning_rate="constant", learning_rate_init=params['learning_rate_init'],
                        max_iter=5000, momentum=0.9, validation_fraction=0.1, shuffle=True)

    mlp.fit(X_train,y_train)
    return mlp