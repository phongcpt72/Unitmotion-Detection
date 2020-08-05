# import numpy as np
# from tensorflow import keras
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# import pandas as pd
#
# sensors = pd.read_csv('DataforPredict/Overlap50windowtime1s.csv')
# feature_cols = ['MaxAx','MaxAy','MaxAz',
#                 'MinAx','MinAy','MinAz',
#                 'MeanAx','MeanAy','MeanAz',
#                 'MedianAx', 'MedianAy', 'MedianAz'
#                 ]
# x = sensors[feature_cols]
# y = sensors.Type
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#
# DTC = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# DTC = DTC.fit(x_train,y_train)
# y_predDTC = DTC.predict(x_test)

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import datasets
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
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
from sklearn.gaussian_process.kernels import  RBF

import warnings
warnings.simplefilter('ignore', FutureWarning)

sensors = pd.read_csv('machine/overlap80_2s/total.csv')
# feature_cols = ['MaxAx','MaxAy','MaxAz',
#                 'MinAx','MinAy','MinAz',
#                 'MeanAx','MeanAy','MeanAz',
#                 'MedianAx', 'MedianAy', 'MedianAz'
#                 ]

# Dynamic + Static Activities
feature_cols = [
                'MaxAx','MaxAy','MaxAz',
                'MaxMx','MaxMy','MaxMz',
                'MaxLiAx','MaxLiAy','MaxLiAz'
                ]

# Static Activities
# feature_cols = [
#                 'MaxAx','MaxAy','MaxAz',
#                 ]


# Dynamic
# feature_cols = [
#                 'MaxAx','MaxAy','MaxAz',
#                 'MaxLiAx','MaxLiAy','MaxLiAz',
#                 'MaxMx','MaxMy','MaxMz'
#                 ]

# #Holding Style
# feature_cols = [
#                 'MaxAx','MaxAy','MaxAz',
#                 ]

# feature_cols = [
#                 'MaxAx','MaxAy','MaxAz',
#                 'MaxGx','MaxGy','MaxGz',
#                 'MedianAx', 'MedianAy', 'MedianAz',
#                 'MedianGx', 'MedianGy', 'MedianGz',
#                 'MedianLiAx','MedianLiAy','MedianLiAz',
#                 'MaxLiAx','MaxLiAy','MaxLiAz',
#                 'MinAx','MinAy','MinAz',
#                 'MinGx','MinGy','MinGz',
#                 'MinLiAx','MinLiAy','MinLiAz',
#                 'MeanMagAcc',
#                 'MeanMagGyr',
#                 'MeanMagMag',
#                 'MeanMagLiAcc',
#                 'MeanAx','MeanAy','MeanAz','MeanGx','MeanGy','MeanGz',
#                 'IqrMagAcc', 'IqrMagGyr',
#                 'IqrMagMag',
#                 'IqrMagLiAcc',
#                 'STDMagAcc',
#                 'STDMagGyr',
#                 'STDMagMag','STDMagLiAcc',
#                 'VarMagAcc',
#                 'VarMagGyr',
#                 'VarMagMag',
#                 'VarMagLiAcc',
#                 'MeanMx','MeanMy','MeanMz','MeanLiAx','MeanLiAy','MeanLiAz',
#                 'MaxMx','MaxMy','MaxMz',
#                 'MedianMx', 'MedianMy', 'MedianMz',
#                 'MinMx','MinMy','MinMz',
#                 'VarianceAx', 'VarianceAy', 'VarianceAz',
#                 'VarianceGx', 'VarianceGy', 'VarianceGz',
#                 'VarianceMx', 'VarianceMy', 'VarianceMz',
#                 'VarianceLiAx','VarianceLiAy','VarianceLiAz',
#                 'StandardDeviationAx','StandardDeviationAy','StandardDeviationAz',
#                 'StandardDeviationGx','StandardDeviationGy','StandardDeviationGz',
#                 'StandardDeviationMx','StandardDeviationMy','StandardDeviationMz',
#                 'StandardDeviationLiAx','StandardDeviationLiAy','StandardDeviationLiAz',
#                 'KurtosisAx', 'KurtosisAy', 'KurtosisAz',
#                 'KurtosisGx', 'KurtosisGy', 'KurtosisGz',
#                 'KurtosisMx', 'KurtosisMy', 'KurtosisMz',
#                 'KurtosisLiAx','KurtosisLiAy','KurtosisLiAz',
#                 'SkeAx', 'SkeAy', 'SkeAz',
#                 'SkeGx', 'SkeGy', 'SkeGz',
#                 'SkeMx', 'SkeMy', 'SkeMz',
#                 'SkeLiAx','SkeLiAy','SkeLiAz',
#                 'SMAAcc', 'SMAGyr',
#                 'SMAMag',
#                 'SMALiAcc'
#                 ]

# feature_cols = [
#     'MaxAx', 'MaxAy', 'MaxAz',
#     'MaxGx', 'MaxGy', 'MaxGz',
#     'MaxMx', 'MaxMy', 'MaxMz'
#     ]

x = sensors[feature_cols]

#y = sensors.Type
#y = sensors.GroupType

#y = sensors.Activities
y = sensors.Unitmotion
#y = sensors.Pose
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#
# SVM = svm.SVC(kernel='linear')
# SVM.fit(x_train, y_train)
# y_predSVM = SVM.predict(x_test)
# print("SVM:",metrics.accuracy_score(y_test, y_predSVM))


start = time.time()
DTC = DecisionTreeClassifier(criterion="entropy", max_depth=3)
DTC = DTC.fit(x_train,y_train)
end = time.time()
# porter = Porter(DTC, language='java') #out ra code tren java
# output = porter.export(embed_data = True)
# print(output)

y_predDTC = DTC.predict(x_test)
print(end-start)
print("DT: ",metrics.accuracy_score(y_test, y_predDTC))
# print(confusion_matrix(y_test,y_predDTC))
# labels = np.unique(y_test)
# a = pd.DataFrame(confusion_matrix(y_test,y_predDTC, labels=labels), index=labels, columns=labels)
# a.to_csv('DT.csv',index=False)
# porter = Porter(DTC, language='java') #out ra code tren java
# output = porter.export(embed_data = True)
# #print(output)
#
# file2write=open("codeDT",'w')
# file2write.write(output)
# file2write.close()


# SVM = svm.SVC(kernel='linear')
# SVM.fit(x_train, y_train)
# y_predSVM = SVM.predict(x_test)
# print("SVM:",metrics.accuracy_score(y_test, y_predSVM))
# #print(confusion_matrix(y_test,y_predSVM))
#

start = time.time()
knn5 = KNeighborsClassifier(n_neighbors=1)
knn5.fit(x_train,y_train)
end = time.time()
y_predknn5 = knn5.predict(x_test)
print(end-start)
print("KNN 3 :",metrics.accuracy_score(y_test, y_predknn5))
#print(confusion_matrix(y_test,y_predknn5))
# labels = np.unique(y_test)
# a = pd.DataFrame(confusion_matrix(y_test,y_predknn5, labels=labels), index=labels, columns=labels)
# a.to_csv('KNN.csv',index=False)
# scores = cross_val_score(knn5, x, y, cv=10, scoring='accuracy')
# print(scores)


# clf = RandomForestClassifier(max_depth= 2, random_state=0)
# clf.fit(x_train,y_train)
# yrfc = clf.predict(x_test)
# print("RFC: ", metrics.accuracy_score(y_test, yrfc))

start = time.time()
clf = ExtraTreesClassifier(n_estimators=10, random_state=0)
clf.fit(x_train,y_train)
end = time.time()
yrfc = clf.predict(x_test)
print(end-start)
print("ETC: ", metrics.accuracy_score(y_test, yrfc))

# porter = Porter(clf, language='java') #out ra code tren java
# output = porter.export(embed_data = True)
# #print(output)
#
# file2write=open("filename",'w')
# file2write.write(output)
# file2write.close()


#print(confusion_matrix(y_test,yrfc, labels=labels))
#print(labels)
# labels = np.unique(y_test)
# a = pd.DataFrame(confusion_matrix(y_test,yrfc, labels=labels), index=labels, columns=labels)
# a.to_csv('ETC.csv',index=False)

# from sklearn.gaussian_process import GaussianProcessClassifier
# start = time.time()
# kernel = 1.0 * RBF(1.0)
# rbf = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(x_train,y_train)
# end = time.time()
# yrfc = clf.predict(x_test)
# print(end-start)
# print("ETC: ", metrics.accuracy_score(y_test, yrfc))

#
#
# clf = AdaBoostClassifier(n_estimators=100, random_state=0)
# clf.fit(x_train,y_train)
# yrfc = clf.predict(x_test)
# print("ABC: ", metrics.accuracy_score(y_test, yrfc))
#
#
# clf = MLPClassifier(random_state=1, max_iter=300)
# clf.fit(x_train,y_train)
# yrfc = clf.predict(x_test)
# print("MLP: ", metrics.accuracy_score(y_test, yrfc))
#
#
# knn10 = KNeighborsClassifier(n_neighbors=10)
# knn10.fit(x_train,y_train)
# y_predknn10 = knn10.predict(x_test)
# print("KNN 10 :",metrics.accuracy_score(y_test, y_predknn10))
# #print(confusion_matrix(y_test,y_predknn10))
#
# NVB = GaussianNB()
# NVB.fit(x_train,y_train)
# y_predNVB = NVB.predict(x_test)
# print("NVB :",metrics.accuracy_score(y_test, y_predNVB))
# #print(confusion_matrix(y_test,y_predNVB))
#
# Ber = BernoulliNB()
# Ber.fit(x_train,y_train)
# y_predBer = Ber.predict(x_test)
# print("Ber :",metrics.accuracy_score(y_test, y_predBer))





# feature_cols = ['MaxAx','MaxAy','MaxAz','MaxGx','MaxGy',
#                 'MaxGz','MaxMx','MaxMy','MaxMz','MaxLiAx','MaxLiAy','MaxLiAz',
#                 'MinAx','MinAy','MinAz','MinGx','MinGy','MinGz',
#                 'MinMx','MinMy','MinMz', 'MinLiAx','MinLiAy','MinLiAz',
#                 'MeanAx','MeanAy','MeanAz','MeanGx','MeanGy','MeanGz',
#                 'MeanMx','MeanMy','MeanMz','MeanLiAx','MeanLiAy','MeanLiAz',
#                 'MedianAx', 'MedianAy', 'MedianAz',
#                 'MedianGx', 'MedianGy', 'MedianGz',
#                 'MedianMx', 'MedianMy', 'MedianMz',
#                 'MedianLiAx','MedianLiAy','MedianLiAz',
#                 'VarianceAx', 'VarianceAy', 'VarianceAz',
#                 'VarianceGx', 'VarianceGy', 'VarianceGz',
#                 'VarianceMx', 'VarianceMy', 'VarianceMz',
#                 'VarianceLiAx','VarianceLiAy','VarianceLiAz',
#                 'StandardDeviationAx','StandardDeviationAy','StandardDeviationAz',
#                 'StandardDeviationGx','StandardDeviationGy','StandardDeviationGz',
#                 'StandardDeviationMx','StandardDeviationMy','StandardDeviationMz',
#                 'StandardDeviationLiAx','StandardDeviationLiAy','StandardDeviationLiAz',
#                 'KurtosisAx', 'KurtosisAy', 'KurtosisAz',
#                 'KurtosisGx', 'KurtosisGy', 'KurtosisGz',
#                 'KurtosisMx', 'KurtosisMy', 'KurtosisMz',
#                 'KurtosisLiAx','KurtosisLiAy','KurtosisLiAz',
#                 'SkeAx', 'SkeAy', 'SkeAz',
#                 'SkeGx', 'SkeGy', 'SkeGz',
#                 'SkeMx', 'SkeMy', 'SkeMz',
#                 'SkeLiAx','SkeLiAy','SkeLiAz',
#                 'SMAAcc', 'SMAGyr', 'SMAMag','SMALiAcc']