import time
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score,precision_recall_curve, auc, roc_auc_score
import pandas as pd
import numpy as np
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
# pos_label=['Downstairs','Grab the phone','Passing the door',
#                                                                              'Phone on table','Put the phone','Stading',
#                                                                              'Upstairs','Walking']
import warnings
warnings.simplefilter('ignore', FutureWarning)


def ETC(x_train, x_test, y_train, y_test, ntrees):
    start = time.time()
    etc = ExtraTreesClassifier(n_estimators=ntrees, random_state=0)
    etc.fit(x_train, y_train)
    end = time.time()
    y_pred_prob = etc.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob, pos_label=[0,1,2,3,4,5,6,7])
    roc_auc = auc(fpr, tpr)
    precision, recall, th_rf = precision_recall_curve(y_test, y_pred_prob)

    return fpr, tpr, roc_auc, precision, recall, end - start

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
    y = sensors.GroupType


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    ETC(x_train, x_test, y_train, y_test,10)