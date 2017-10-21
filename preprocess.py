import pandas as pd
import logging
import numpy as np
logging.basicConfig(
    format='%(asctime)s :%(levelname)s : %(message)s',
    level=logging.INFO
)

"""
* Divide the data into Train and Test two part.
    Train set for machine learning and deep learning 
    Test set for compare the model score with other statistic method
* Expect:
    -- save = 'yes' or 'no'. 'yes' will save the .csv into dir. 'no' will return the X_train, Y_train, X_test, Y_test as np.array
* Return:
    -- file 
    -- or X_train, Y_train, X_test, Y_test
"""
def get_data(save = 'no'):
# Load data
    logging.info('Loading data ......')
    data = pd.read_csv('data/origin_data.csv').iloc[:,1:]
    label = data['RACE']
    del data['RACE']
    logging.info('Get data and label')

    X_train = data.iloc[:210000,:]
    X_test = data.iloc[210000:300000,:]

    Y_train = pd.DataFrame(label.iloc[:210000])
    Y_test = pd.DataFrame(label.iloc[210000:300000])
    if save == 'yes':
        X_train.to_csv('data/X_train.csv',index=False)
        X_test.to_csv('data/For test/X_test.csv',index=False)

        Y_train.to_csv('data/Y_train.csv',index=False)
        Y_test.to_csv('data/For test/Y_test.csv',index=False)
    if save == 'no':
        del X_train['KEY']
        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

"""
* Clean the data, keep continuous, set the catagorical data into dummy  
* Expect:
    -- X = feature data
    -- y = label data
* Return:
    -- dummy feature set
"""
def clean_data(X,y):
    logging.info('start Dummy ......')
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    train_dummy = pd.DataFrame()
    undummy_index = [0,6,10,11,12,13,14,18,24,54,55]
    for i in range(X.shape[1]):
        tmp_column = X.iloc[:, i]
        if i in undummy_index:
            train_dummy = pd.concat([train_dummy, tmp_column], axis=1)
        else:
            tmp_dummy = pd.get_dummies(tmp_column)
            train_dummy = pd.concat([train_dummy, tmp_dummy], axis=1)
    logging.info('Dummy finish ......')
    label_dummy = pd.get_dummies(y.iloc[:,0])
    return train_dummy, label_dummy

def get_dummy():
    X, y, _, _ = get_data('no')
    feature, label = clean_data(X,y)
    return feature,label,X, y
if __name__ == '__main__':
    get_data('yes')
    # feature, label = clean_data(X,y)
