# Load the package
import utils
import os
import pickle
import logging
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

def ada_train(X_train, y_train, indices):
    # get the result of Ada boost 5 cv using 5 grid search to get 5 models
    if not os.path.exists('cache/ada.pkl'):
        logging.info('No ada para found, Doing Grid-search')
        ada_models = []
        for i in range(5):
            logging.info('Ada Grid search for {} validation ......'.format(i))
            X_cvtrain, y_cvtrain, _, _ = utils.get_each_set(i, indices, X_train, y_train)
            # params = utils.Ada_para_search(X_train,y_train)
            # Todo: above function
            # Ada_model = utils.train_ada(params, X_train, y_train)
            ada_models.append(Ada_model)
        ada_models_pkl = open('cache/ada.pkl', 'wb')
        pickle.dumps(ada_models,ada_models_pkl)
        ada_models_pkl.close()
        logging.info('Finish saving ada models')
    else:
        logging.info('Already did the Ada grid search before......')
        logging.info('Loading params .......')
        models = open('cache/ada.pkl', 'rb')
        ada_models = pickle.load(models)

    logging.info('Get the results of Ada ......')
    ada_result = utils.stacking(ada_models,indices,X_train)
    return ada_result
# get the result of Elastic net


# results of SGD Classfier


# KNN


# Gaussian Process Classification (GPC)

# Multinomial Naive Bayes

# MLP


# tensorflow

# Stacking all of it
# first layer

# second layer

# out put and show the ROC

if __name__ == '__main__':
    X_train, y_train, X_test, y_test, X_miss, y_miss, index_data, indices = first_step()
    ada_result = ada_train(X_train, y_train, indices)