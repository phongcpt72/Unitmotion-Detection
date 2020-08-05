import numpy as np
import time
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score,precision_recall_curve, auc, roc_auc_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB , BernoulliNB
from sklearn_porter import Porter
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore', FutureWarning)


def DTC(x_train, x_test, y_train, y_test):
    start = time.time()
    dtc = DecisionTreeClassifier(criterion="entropy", max_depth=1)
    dtc.fit(x_train, y_train)
    end = time.time()
    y_pred_prob = dtc.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, th_rf = precision_recall_curve(y_test, y_pred_prob)

    return fpr, tpr, roc_auc, precision, recall, end - start

def ETC(x_train, x_test, y_train, y_test, ntrees):
    start = time.time()
    etc = ExtraTreesClassifier(n_estimators=ntrees, random_state=0)
    etc.fit(x_train, y_train)
    end = time.time()
    y_pred_prob = etc.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, th_rf = precision_recall_curve(y_test, y_pred_prob)

    return fpr, tpr, roc_auc, precision, recall, end - start

def KNN(x_train, x_test, y_train, y_test, k):
    start = time.time()
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn = knn.fit(x_train, y_train)
    end = time.time()
    y_pred_prob = knn.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, th_rf = precision_recall_curve(y_test, y_pred_prob)

    return fpr, tpr, roc_auc, precision, recall, end - start

def RFC(x_train, x_test, y_train, y_test, ntrees):
    start = time.time()
    rfc = RandomForestClassifier(n_estimators= ntrees, random_state=0)
    rfc.fit(x_train, y_train)
    end = time.time()
    y_pred_prob = rfc.predict_proba(x_test)[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr,tpr)
    precision, recall, th_rf = precision_recall_curve(y_test, y_pred_prob)

    return fpr, tpr, roc_auc,precision,recall, end - start

if __name__ == '__main__':
    # featols = [
    #                 'MaxAx','MaxAy','MaxAz'
    #                 ]
    # feature_cols = [
    #                     'MaxAx','MaxAy','MaxAz',
    #                     'MaxLiAx','MaxLiAy','MaxLiAz',
    #                     'MaxMx','MaxMy','MaxMz'
    #                 ]
    # feature_cols = [
    #                 'MaxAx','MaxAy','MaxAz',
    #                 'MaxMx','MaxMy','MaxMz',
    #                 'MaxLiAx','MaxLiAy','MaxLiAz'
    #                 ]

    #tong hop ko tach luon ca tach rieng
    feature_cols = [
    'MaxAx', 'MaxAy', 'MaxAz',
    'MaxGx', 'MaxGy', 'MaxGz',
    'MaxMx', 'MaxMy', 'MaxMz'
    ]


    filename = 'total'
    filecur  = 'overlap80_2s'
    sensors = pd.read_csv(f'machine/{filecur}/{filename}.csv')
    savefile = (f'machine/{filecur}/ML/{filename}ml.csv')

    x = sensors[feature_cols]
    y = sensors.Unitmotion


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    fprdt, tprdt, roc_aucdt, precisiondt, recalldt, timedt = DTC(x_train, x_test, y_train, y_test)
    fprknn3, tprknn3, roc_aucknn3, precisionknn3, recallknn3, timeknn3 = KNN(x_train, x_test, y_train, y_test,3)
    fprknn5, tprknn5, roc_aucknn5, precisionknn5, recallknn5, timeknn5 = KNN(x_train, x_test, y_train, y_test,5)
    fpretc100, tpretc100, roc_aucetc100, precisionetc100, recalletc100, timeetc100 = ETC(x_train, x_test, y_train, y_test,100)
    fprrfc100, tprrfc100, roc_aucrfc100, precisionrfc100, recallrfc100, timerfc100 = RFC(x_train, x_test, y_train, y_test,100)
    #fpr, tpr, roc_auc, precision, recall, time


    plt.figure(1)
    plt.plot([0,1],[0,1], 'k--')
    plt.plot(fprdt, tprdt, 'DT (area = %0.3f)' % roc_aucdt)
    plt.plot(fprknn3, tprknn3, 'KNN3 (area = %0.3f)' % roc_aucknn3)
    plt.plot(fprknn5, tprknn5, 'KNN5 (area = %0.3f)' % roc_aucknn5)
    plt.plot(fpretc100, tpretc100, 'ETC (area = %0.3f)' % roc_aucetc100)
    plt.plot(fprrfc100, tprrfc100, 'RFC (area = %0.3f)' % roc_aucrfc100)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves from the investigated models')
    plt.legend(loc='best')
    plt.show()