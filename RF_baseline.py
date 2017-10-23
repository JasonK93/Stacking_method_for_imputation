import numpy as np
from sklearn import datasets,cross_validation,ensemble
import pandas as pd
from sklearn.model_selection import train_test_split
import preprocess


from scipy import interp
import matplotlib as mpl; mpl.use('Agg') # do not run X server
import matplotlib.pyplot as plt; plt.style.use('ggplot')
from sklearn.metrics import roc_curve, auc,confusion_matrix
from sklearn.preprocessing import label_binarize
seed = 520

def compute_roc(y_test, y_test_proba, nb_classes):
    y_test = label_binarize(y_test, classes=range(0, nb_classes))
    y_test_proba = label_binarize(y_test_proba, classes=range(0, nb_classes))
    print(y_test[0],y_test_proba[0])
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_test_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(),
        y_test_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc, fpr, tpr

''' 
* Creates and saves ROC plots to file.
* Expects:
    - roc_auc = dictionary of ROC_AUC values
    - fpr = dictionary of false positive rates
    - tpr = dictionary of true positive rate
    - nb_classes = number of classes
    - path = string filepath describing where to save files
'''
# Plot ROC curves for the multiclass problem
def save_plots(roc_auc, fpr, tpr, nb_classes):
    # aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

    # interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # average and compute AUC
    mean_tpr /= nb_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # plot
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
        label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
        linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"],
        label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
        linewidth=2)
    for i in range(nb_classes):
        plt.plot(fpr[i], tpr[i],
            label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")

    # plt.savefig(path) # save plot
    plt.show()
    # plt.close()


def test_RandomForestClassifier(*data):
    '''
    test the RF method
    :param data: train_data, test_data, train_value, test_value
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    clf=ensemble.RandomForestClassifier(n_estimators=100, max_features='sqrt')
    clf.fit(X_train,y_train)
    print("Traing Score:%f"%clf.score(X_train,y_train))
    print("Testing Score:%f"%clf.score(X_test,y_test))
    print(confusion_matrix(y_test,clf.predict(X_test)))
    roc_auc, fpr, tpr = compute_roc(y_test,clf.predict(X_test),6)
    save_plots(roc_auc, fpr, tpr, 6)

if __name__ == '__main__':

    print('Loading data ......')
    X = pd.read_csv('data/train.csv').iloc[:,1:]
    y = X['RACE']
    del X['RACE']
    del X['KEY']
    X = np.array(X)
    y = np.array(y).ravel()
    y = y - 1

    # X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.75, test_size=0.25, random_state=seed)

    X_test = pd.read_csv('data/ctest.csv').iloc[:,1:]
    y_test = X_test['RACE']
    del X_test['RACE']
    del X_test['KEY']
    X_test = np.array(X_test)
    y_test = np.array(y_test).ravel()
    y_test = y_test - 1
    X_train = X
    y_train = y
    print('finish loading .....')
    test_RandomForestClassifier(X_train,X_test,y_train,y_test)

    print('missing acc: ......')
    index_data = np.isnan(np.array(pd.read_csv('data/test.csv')['RACE']))
    test_RandomForestClassifier(X_train,X_test[index_data],y_train,y_test[index_data])
    """
    Traing Score:0.999989
    Testing Score:0.846001
    [[156645   3104    523     35      0    236]
     [ 14194  27867   2664     88      0    254]
     [  6172   2001  11628      2      0     72]
     [  1288    357     39    279      0     59]
     [   281     62     54      1      5      2]
     [  3422   1348    212    121      0   4590]]
    (array([1, 0, 0, 0, 0, 0]), array([1, 0, 0, 0, 0, 0]))
    missing acc: ......
    Traing Score:0.999992
    Testing Score:0.837511
    [[41162   823   196     5     0    64]
     [ 3044  5344  1006    13     0    48]
     [ 2118   819  4171     1     0    20]
     [  292    45    10    24     0    15]
     [  133    27    40     1     3     1]
     [  858   351    83     9     0   952]]
    (array([1, 0, 0, 0, 0, 0]), array([1, 0, 0, 0, 0, 0])
    """

    print('Starting grid_search....')
