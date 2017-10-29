# Load the package
import utils
import os
import pickle
import logging
import numpy as np
import RF_baseline
logging.basicConfig(
    format='%(asctime)s :%(levelname)s : %(message)s',
    level=logging.INFO
)


def first_step():
    # Load the data
    logging.info('Loading Data ......')
    X_train, y_train, X_test, y_test, X_miss, y_miss, index_data = utils.get_data()

    # Load the CV indices
    logging.info('get the K-fold indices ......')
    indices = utils.get_kfold_indices(X_train)

    # Set up a directory for model parameters
    if not os.path.exists('cache'):
        logging.info('Create cache dir .......')
        os.mkdir('cache')
    else:
        logging.info('already exist cache dir ......')
    return X_train, y_train, X_test, y_test, X_miss, y_miss, index_data, indices


# Ada-Boosting based DT
# best score: 0.229886990004
def ada_train(X_train, y_train, indices):
    # get the result of Ada boost 5 cv using 5 grid search to get 5 models
    if not os.path.exists('cache/ada.pkl'):
        logging.info('No ada para found, Doing Grid-search')
        ada_models = []
        for i in range(5):
            logging.info('Ada Grid search for {} validation ......'.format(i))
            X_cvtrain, y_cvtrain, _, _ = utils.get_each_set(i, indices, X_train, y_train)
            params = utils.Ada_para_search(X_cvtrain,y_cvtrain)
            # Todo: above function
            Ada_model = utils.train_ada(params, X_cvtrain, y_cvtrain)
            ada_models.append(Ada_model)
        ada_models_pkl = open('cache/ada.pkl', 'wb')
        pickle.dump(ada_models, ada_models_pkl)
        ada_models_pkl.close()
        logging.info('Finish saving ada models')
    else:
        logging.info('Already did the Ada grid search before......')
        logging.info('Loading params .......')
        models = open('cache/ada.pkl', 'rb')
        ada_models = pickle.load(models)

    logging.info('Get the results of Ada ......')
    ada_result = utils.stacking(ada_models, indices, X_train)
    return ada_result


# get the result of Elastic net
# best score: -0.0277490303477
def elastic_train(X_train, y_train, indices):
    if not os.path.exists('cache/elastic.pkl'):
        logging.info('No elastic para found, Doing Grid-search')
        elastic_models = []
        for i in range(5):
            logging.info('Elastic net Grid search for {} validation ......'.format(i))
            X_cvtrain, y_cvtrain, _, _ = utils.get_each_set(i, indices, X_train, y_train)
            params = utils.Elastic_para_search(X_cvtrain,y_cvtrain)
            # Todo: above function
            elastic_model = utils.train_elastic(params, X_cvtrain, y_cvtrain)
            elastic_models.append(elastic_model)
        elastic_models_pkl = open('cache/elastic.pkl', 'wb')
        pickle.dump(elastic_models, elastic_models_pkl)
        elastic_models_pkl.close()
        logging.info('Finish saving elastic models')
    else:
        logging.info('Already did the Elastic Net grid search before......')
        logging.info('Loading params .......')
        models = open('cache/elastic.pkl', 'rb')
        elastic_models = pickle.load(models)

    logging.info('Get the results of Elastic Net ......')
    elastic_result = utils.stacking(elastic_models, indices, X_train)
    return elastic_result


# results of SGD Classfier
# best score: 0.675857794657
def sgd_train(X_train, y_train, indices):
    if not os.path.exists('cache/sgd.pkl'):
        logging.info('No sgd para found, Doing Grid-search')
        sgd_models = []
        for i in range(5):
            logging.info('sgd Grid search for {} validation ......'.format(i))
            X_cvtrain, y_cvtrain, _, _ = utils.get_each_set(i, indices, X_train, y_train)
            params = utils.sgd_para_search(X_cvtrain,y_cvtrain)
            # Todo: above function
            sgd_model = utils.train_sgd(params, X_cvtrain, y_cvtrain)
            sgd_models.append(sgd_model)
        sgd_models_pkl = open('cache/sgd.pkl', 'wb')
        pickle.dump(sgd_models, sgd_models_pkl)
        sgd_models_pkl.close()
        logging.info('Finish saving sgd models')
    else:
        logging.info('Already did the SGD grid search before......')
        logging.info('Loading params .......')
        models = open('cache/sgd.pkl', 'rb')
        sgd_models = pickle.load(models)

    logging.info('Get the results of SGD ......')
    sgd_result = utils.stacking(sgd_models, indices, X_train)
    return sgd_result


# KNN
def knn_train(X_train, y_train, indices):
    if not os.path.exists('cache/knn.pkl'):
        logging.info('No knn para found, Doing Grid-search')
        knn_models = []
        for i in range(5):
            logging.info('knn Grid search for {} validation ......'.format(i))
            X_cvtrain, y_cvtrain, _, _ = utils.get_each_set(i, indices, X_train, y_train)
            params = utils.knn_para_search(X_cvtrain,y_cvtrain)
            # Todo: above function
            knn_model = utils.train_knn(params, X_cvtrain, y_cvtrain)
            knn_models.append(knn_model)
        knn_models_pkl = open('cache/knn.pkl', 'wb')
        pickle.dump(knn_models, knn_models_pkl)
        knn_models_pkl.close()
        logging.info('Finish saving knn models')
    else:
        logging.info('Already did the KNN grid search before......')
        logging.info('Loading params .......')
        models = open('cache/knn.pkl', 'rb')
        knn_models = pickle.load(models)

    logging.info('Get the results of KNN ......')
    knn_result = utils.stacking(knn_models, indices, X_train)
    return knn_result


# Todo: more kernel need try on this side
# Gaussian Process Classification (GPC)
def gpc_train(X_train, y_train, indices):
    if not os.path.exists('cache/gpc.pkl'):
        logging.info('No gpc para found, Doing Grid-search')
        gpc_models = []
        for i in range(5):
            logging.info('gpc Grid search for {} validation ......'.format(i))
            X_cvtrain, y_cvtrain, _, _ = utils.get_each_set(i, indices, X_train, y_train)
            params = utils.gpc_para_search(X_cvtrain,y_cvtrain)
            # Todo: above function
            gpc_model = utils.train_gpc(params, X_cvtrain, y_cvtrain)
            gpc_models.append(gpc_model)
        gpc_models_pkl = open('cache/gpc.pkl', 'wb')
        pickle.dump(gpc_models, gpc_models_pkl)
        gpc_models_pkl.close()
        logging.info('Finish saving gpc models')
    else:
        logging.info('Already did the GPC grid search before......')
        logging.info('Loading params .......')
        models = open('cache/gpc.pkl', 'rb')
        gpc_models = pickle.load(models)

    logging.info('Get the results of GPC ......')
    gpc_result = utils.stacking(gpc_models, indices, X_train)
    return gpc_result


# Todo: more NB method can be used here
# Multinomial Naive Bayes
def mnb_train(X_train, y_train, indices):
    if not os.path.exists('cache/mnb.pkl'):
        logging.info('No mnb para found, Doing Grid-search')
        mnb_models = []
        for i in range(5):
            logging.info('mnb Grid search for {} validation ......'.format(i))
            X_cvtrain, y_cvtrain, _, _ = utils.get_each_set(i, indices, X_train, y_train)
            params = utils.mnb_para_search(X_cvtrain,y_cvtrain)
            # Todo: above function
            mnb_model = utils.train_mnb(params, X_cvtrain, y_cvtrain)
            mnb_models.append(mnb_model)
        mnb_models_pkl = open('cache/mnb.pkl', 'wb')
        pickle.dump(mnb_models, mnb_models_pkl)
        mnb_models_pkl.close()
        logging.info('Finish saving mnb models')
    else:
        logging.info('Already did the MNB grid search before......')
        logging.info('Loading params .......')
        models = open('cache/mnb.pkl', 'rb')
        mnb_models = pickle.load(models)

    logging.info('Get the results of MNB ......')
    mnb_result = utils.stacking(mnb_models, indices, X_train)
    return mnb_result

# RF
"best score: 0.274526351879"
def rf_train(X_train, y_train, indices):
    if not os.path.exists('cache/rf.pkl'):
        logging.info('No rf para found , Doing grid-search ......')
        models = []
        for i in range(5):
            logging.info('Grid search for the {} validation ......'.format(i))
            X_cvtrain, y_cvtrain, _, _ = utils.get_each_set(i, indices, X_train, y_train)
            params = utils.RF_para_search(X_cvtrain, y_cvtrain)
            RF_model1 = utils.train_RF(params, X_cvtrain, y_cvtrain)
            models.append(RF_model1)
        models_pkl = open('cache/rf.pkl', 'wb')
        pickle.dump(models, models_pkl)
        models_pkl.close()
        logging.info('Finish saving the parameters')
    else:
        logging.info('already did the grid search ....load parameters ....')
        fr = open('cache/rf.pkl', 'rb')
        models = pickle.load(fr)

    rf_result = utils.stacking(models, indices, X_train)
    return rf_result

# MLP
def mlp_train(X_train, y_train, indices):
    if not os.path.exists('cache/mlp.pkl'):
        logging.info('No mlp para found, Doing Grid-search')
        mlp_models = []
        for i in range(5):
            logging.info('mlp Grid search for {} validation ......'.format(i))
            X_cvtrain, y_cvtrain, _, _ = utils.get_each_set(i, indices, X_train, y_train)
            params = utils.mlp_para_search(X_cvtrain,y_cvtrain)
            # Todo: above function
            mlp_model = utils.train_mlp(params, X_cvtrain, y_cvtrain)
            mlp_models.append(mlp_model)
        mlp_models_pkl = open('cache/mlp.pkl', 'wb')
        pickle.dump(mlp_models, mlp_models_pkl)
        mlp_models_pkl.close()
        logging.info('Finish saving mlp models')
    else:
        logging.info('Already did the MLP grid search before......')
        logging.info('Loading params .......')
        models = open('cache/mlp.pkl', 'rb')
        mlp_models = pickle.load(models)

    logging.info('Get the results of MLP ......')
    mlp_result = utils.stacking(mlp_models, indices, X_train)
    return mlp_result

# tensorflow

# Stacking all of it
# first layer

# second layer

# out put and show the ROC

def cal():
    X_train, y_train, X_test, y_test, X_miss, y_miss, index_data, indices = first_step()
    ada_result = ada_train(X_train, y_train, indices) # shape (379171, 5)
    elastic_result = elastic_train(X_train, y_train, indices)
    sgd_result = sgd_train(X_train, y_train, indices)
    knn_result = knn_train(X_train, y_train, indices)
    gpc_result = gpc_train(X_train, y_train, indices)
    mnb_result = mnb_train(X_train, y_train, indices)
    mlp_result = mlp_train(X_train, y_train, indices)
    rf_result = rf_train(X_train, y_train, indices)
    stacking_layer1 = np.concatenate([ada_result, elastic_result, sgd_result,
                                      mnb_result, mlp_result, rf_result],axis=1)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(stacking_layer1[0:250000],y_train[0:250000])
    print("Traing Score:%f" % clf.score(stacking_layer1[0:250000], y_train[0:250000]))
    print("Testing Score:%f" % clf.score(stacking_layer1[250000:], y_train[250000:]))
    print(confusion_matrix(y_train[250000:], clf.predict(X_train[250000:])))
    roc_auc, fpr, tpr = RF_baseline.compute_roc(y_train[250000:], clf.predict(X_train[250000:]), 6)
    RF_baseline.save_plots(roc_auc, fpr, tpr, 6)



if __name__ == '__main__':
    X_train, y_train, X_test, y_test, X_miss, y_miss, index_data, indices = first_step()
    ada_result = ada_train(X_train, y_train, indices) # shape (379171, 5)
    elastic_result = elastic_train(X_train, y_train, indices)
    sgd_result = sgd_train(X_train, y_train, indices)
    # knn_result = knn_train(X_train, y_train, indices)
    # gpc_result = gpc_train(X_train, y_train, indices)
    # mnb_result = mnb_train(X_train, y_train, indices)
    mlp_result = mlp_train(X_train, y_train, indices)
    rf_result = rf_train(X_train, y_train, indices)
    stacking_layer1 = np.concatenate([ada_result, elastic_result, sgd_result,
                                      mlp_result, rf_result],axis=1)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(stacking_layer1[0:250000],y_train[0:250000])
    print("Traing Score:%f" % clf.score(stacking_layer1[0:250000], y_train[0:250000]))
    print("Testing Score:%f" % clf.score(stacking_layer1[250000:], y_train[250000:]))
    print(confusion_matrix(y_train[250000:], clf.predict(X_train[250000:])))
    roc_auc, fpr, tpr = RF_baseline.compute_roc(y_train[250000:], clf.predict(X_train[250000:]), 6)
    RF_baseline.save_plots(roc_auc, fpr, tpr, 6)
