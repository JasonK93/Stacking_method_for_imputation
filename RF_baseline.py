import matplotlib.pyplot as plt
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

X, _, _, y = preprocess.get_dummy()
y = y - 1
print('finish loading .....')
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.75, test_size=0.25, random_state=seed)

test_RandomForestClassifier(X_train,X_test,y_train,y_test)


"""
macro:0.69
Traing Score:0.998933
Testing Score:0.868857
[[35698   799    70     3     0    64]
 [ 2609  7176    67    11     0    47]
 [  869   161  1478     1     0    28]
 [  549   240    18   198     0    39]
 [   52     9     4     1     0     0]
 [  862   326    16    40     0  1065]]
[1 0 0 0 0 0] [1 0 0 0 0 0]
"""



_, _, X, y = preprocess.get_dummy()
y = y - 1
print('finish loading .....')
X_train = X
y_train = y
# X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.75, test_size=0.25)
X_test = pd.read_csv('data/For test/X_test.csv')
del X_test['KEY']
y_test = pd.read_csv('data/For test/Y_test.csv')
y_test = y_test - 1
test_RandomForestClassifier(X_train,X_test,y_train,y_test)


"""
macro: 0.56

Traing Score:0.999990
Testing Score:0.744678
[[62444  3647   320    24     0    53]
 [ 9360  2386   868    27     0     5]
 [ 3525   921  2075     0     0     4]
 [  957    45   109    65     0    14]
 [   51     4     7     0     0     0]
 [ 2392   284   284    78     0    51]]
"""