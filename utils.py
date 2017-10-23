import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
seed = 520

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
    print(params)
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