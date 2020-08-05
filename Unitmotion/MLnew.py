import numpy as np
import time
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
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

import warnings
warnings.simplefilter('ignore', FutureWarning)

def KNN(x_train, x_test, y_train, y_test, k):
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn = knn.fit(x_train, y_train)
    end = time.time()
    y_predknn = knn.predict(x_test)
    accu = metrics.accuracy_score(y_test, y_predknn)

    # labels = np.unique(y_test)
    # a = pd.DataFrame(confusion_matrix(y_test, y_predknn, labels=labels), index=labels, columns=labels)
    # a.to_csv(f'machine/{filecur}/{filename}KNN{k}.csv', index=False)
    #
    # porter = Porter(knn, language='java')  # out ra code tren java
    # output = porter.export(embed_data=True)
    # file2write = open(f"machine/{filecur}/{filename}codeKNN{k}", 'w')
    # file2write.write(output)
    # file2write.close()

    return end-start, accu, knn

def DTC(x_train, x_test, y_train, y_test):
    start = time.time()
    DTC = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    DTC = DTC.fit(x_train, y_train)
    end = time.time()
    y_predDTC = DTC.predict(x_test)
    accu = metrics.accuracy_score(y_test, y_predDTC)

    # labels = np.unique(y_test)
    # a = pd.DataFrame(confusion_matrix(y_test, y_predDTC, labels=labels), index=labels, columns=labels)
    # a.to_csv(f'machine/{filecur}/{filename}DT.csv', index=False)
    #
    # porter = Porter(DTC, language='java')  # out ra code tren java
    # output = porter.export(embed_data=True)
    # file2write = open(f"machine/{filecur}/{filename}codeDTC", 'w')
    # file2write.write(output)
    # file2write.close()

    return end-start, accu, DTC

def ETC(x_train, x_test, y_train, y_test, ntrees):
    start = time.time()
    etc = ExtraTreesClassifier(n_estimators=ntrees, random_state=0)
    etc = etc.fit(x_train, y_train)
    end = time.time()
    yETC = etc.predict(x_test)
    accu = metrics.accuracy_score(y_test, yETC)

    # labels = np.unique(y_test)
    # a = pd.DataFrame(confusion_matrix(y_test, yETC, labels=labels), index=labels, columns=labels)
    # a.to_csv(f'machine/{filecur}/{filename}ETC{ntrees}.csv', index=False)
    #
    # porter = Porter(etc, language='java')  # out ra code tren java
    # output = porter.export(embed_data=True)
    # file2write = open(f"machine/{filecur}/{filename}codeETC{ntrees}", 'w')
    # file2write.write(output)
    # file2write.close()

    return end-start, accu, etc

def DTCresult(x_train, x_test, y_train, y_test):
    start = time.time()
    DTC = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    DTC = DTC.fit(x_train, y_train)
    end = time.time()
    y_predDTC = DTC.predict(x_test)
    accu = round(metrics.accuracy_score(y_test, y_predDTC),2)
    pre  = round(precision_score(y_test, y_predDTC, average="macro"),2)
    f1 = round(f1_score(y_test, y_predDTC, average="macro"),2)
    recal = round(recall_score(y_test, y_predDTC, average="macro"),2)

    return accu, pre, f1, recal

def ETCresult(x_train, x_test, y_train, y_test, ntrees):
    start = time.time()
    etc = ExtraTreesClassifier(n_estimators=ntrees, random_state=0)
    etc = etc.fit(x_train, y_train)
    end = time.time()
    yETC = etc.predict(x_test)
    accu = round(metrics.accuracy_score(y_test, yETC),2)
    pre = round(precision_score(y_test, yETC, average="macro"),2)
    f1 = round(f1_score(y_test, yETC, average="macro"),2)
    recal = round(recall_score(y_test, yETC, average="macro"),2)

    return accu, pre, f1, recal

def KNNresult(x_train, x_test, y_train, y_test, k):
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn = knn.fit(x_train, y_train)
    end = time.time()
    y_predknn = knn.predict(x_test)
    accu = round(metrics.accuracy_score(y_test, y_predknn),2)
    pre = round(precision_score(y_test, y_predknn, average="macro"),2)
    f1 = round(f1_score(y_test, y_predknn, average="macro"),2)
    recal = round(recall_score(y_test, y_predknn, average="macro"),2)

    return accu, pre, f1, recal



if __name__ == '__main__':
    # feature_cols = [
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

    if filename == 'StaticDynamic' or filename == 'total':
        x = sensors[feature_cols]
        y = sensors.Activities
    elif filename == 'HoldingStyle':
        x = sensors[feature_cols]
        y = sensors.Pose
    else:
        x = sensors[feature_cols]
        y = sensors.Unitmotion

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    #Decision Tree
    runtime, score, dtc = DTC(x_train, x_test, y_train, y_test)
    print('DTC')
    print(f'Accuracy: {score}, Run time: {runtime}')
    scores = cross_val_score(dtc, x, y, cv=3, scoring='accuracy')
    print(f'K-fold Cross Validation 3 : {scores.mean()}')
    scores = cross_val_score(dtc, x, y, cv=5, scoring='accuracy')
    print(f'K-fold Cross Validation 5 : {scores.mean()}')

    #KNN 3
    k = 3
    runtime, score, knn = KNN(x_train, x_test, y_train, y_test,k)
    print(f'Knn {k}')
    print(f'Accuracy: {score}, Run time: {runtime}')
    scores = cross_val_score(knn, x, y, cv=3, scoring='accuracy')
    print(f'K-fold Cross Validation 3 : {scores.mean()}')
    scores = cross_val_score(knn, x, y, cv=5, scoring='accuracy')
    print(f'K-fold Cross Validation 5 : {scores.mean()}')

    #KNN 5
    k = 5
    runtime, score, knn = KNN(x_train, x_test, y_train, y_test, k)
    print(f'Knn {k}')
    print(f'Accuracy: {score}, Run time: {runtime}')
    scores = cross_val_score(knn, x, y, cv=3, scoring='accuracy')
    print(f'K-fold Cross Validation 3 : {scores.mean()}')
    scores = cross_val_score(knn, x, y, cv=5, scoring='accuracy')
    print(f'K-fold Cross Validation 5 : {scores.mean()}')

    #Extra Trees Classifier with 100 is the number of trees in the forest.
    ntrees= 100
    runtime, score, etc = ETC(x_train, x_test, y_train, y_test, ntrees)
    print(f'ETC {ntrees}')
    print(f'Accuracy: {score}, Run time: {runtime}')
    scores = cross_val_score(etc, x, y, cv=3, scoring='accuracy')
    print(f'K-fold Cross Validation 3 : {scores.mean()}')
    scores = cross_val_score(etc, x, y, cv=5, scoring='accuracy')
    print(f'K-fold Cross Validation 5 : {scores.mean()}')

    # Extra Trees Classifier with 50 is the number of trees in the forest.
    ntrees = 50
    runtime, score, etc = ETC(x_train, x_test, y_train, y_test, ntrees)
    print(f'ETC {ntrees}')
    print(f'Accuracy: {score}, Run time: {runtime}')
    scores = cross_val_score(etc, x, y, cv=3, scoring='accuracy')
    print(f'K-fold Cross Validation 3 : {scores.mean()}')
    scores = cross_val_score(etc, x, y, cv=5, scoring='accuracy')
    print(f'K-fold Cross Validation 5 : {scores.mean()}')

    # Extra Trees Classifier with 25 is the number of trees in the forest.
    ntrees = 25
    runtime, score, etc = ETC(x_train, x_test, y_train, y_test, ntrees)
    print(f'ETC {ntrees}')
    print(f'Accuracy: {score}, Run time: {runtime}')
    scores = cross_val_score(etc, x, y, cv=3, scoring='accuracy')
    print(f'K-fold Cross Validation 3 : {scores.mean()}')
    scores = cross_val_score(etc, x, y, cv=5, scoring='accuracy')
    print(f'K-fold Cross Validation 5 : {scores.mean()}')

    # Extra Trees Classifier with 10 is the number of trees in the forest.
    ntrees = 10
    runtime, score, etc = ETC(x_train, x_test, y_train, y_test, ntrees)
    print(f'ETC {ntrees}')
    print(f'Accuracy: {score}, Run time: {runtime}')
    scores = cross_val_score(etc, x, y, cv=3, scoring='accuracy')
    print(f'K-fold Cross Validation 3 : {scores.mean()}')
    scores = cross_val_score(etc, x, y, cv=5, scoring='accuracy')
    print(f'K-fold Cross Validation 5 : {scores.mean()}')

    accu, pre, f1, recal = DTCresult(x_train, x_test, y_train, y_test)
    print('DTC')
    print(f'Accuracy: {accu}, Recall: {recal}, Precision: {pre}, F1: {f1}')

    accu, pre, f1, recal = KNNresult(x_train, x_test, y_train, y_test,3)
    print('KNN 3')
    print(f'Accuracy: {accu}, Recall: {recal}, Precision: {pre}, F1: {f1}')

    accu, pre, f1, recal = KNNresult(x_train, x_test, y_train, y_test, 5)
    print('KNN 5')
    print(f'Accuracy: {accu}, Recall: {recal}, Precision: {pre}, F1: {f1}')



    accu, pre, f1, recal = ETCresult(x_train, x_test, y_train, y_test, 100)
    print('ETC 100')
    print(f'Accuracy: {accu}, Recall: {recal}, Precision: {pre}, F1: {f1}')

    accu, pre, f1, recal = ETCresult(x_train, x_test, y_train, y_test, 50)
    print('ETC 50')
    print(f'Accuracy: {accu}, Recall: {recal}, Precision: {pre}, F1: {f1}')

    accu, pre, f1, recal = ETCresult(x_train, x_test, y_train, y_test, 25)
    print('ETC 25')
    print(f'Accuracy: {accu}, Recall: {recal}, Precision: {pre}, F1: {f1}')

    accu, pre, f1, recal = ETCresult(x_train, x_test, y_train, y_test, 10)
    print('ETC 10')
    print(f'Accuracy: {accu}, Recall: {recal}, Precision: {pre}, F1: {f1}')