import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB , BernoulliNB
from sklearn_porter import Porter
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import  RBF
import numpy as np
import warnings
warnings.simplefilter('ignore', FutureWarning)

sensors = pd.read_csv('machine/overlap80_2s/newgroup317/HoldingStyles.csv')
# feature_cols = ['MaxAx','MaxAy','MaxAz',
#                 'MinAx','MinAy','MinAz',
#                 'MeanAx','MeanAy','MeanAz',
#                 'MedianAx', 'MedianAy', 'MedianAz'
#                 ]


fnamearr = ['Max','Min','Mean','StandardDeviation']
for i in range(len(fnamearr)):
    fname = fnamearr[i]
    print("\n")
    print(fname)
# NewGroup
    feature_cols = [
                    f'{fname}Ax',f'{fname}Ay',f'{fname}Az',
                    # f'{fname}Gx',f'{fname}Gy',f'{fname}Gz',
                    # f'{fname}Mx',f'{fname}My',f'{fname}Mz'
                    ]

    x = sensors[feature_cols]
    #y = sensors.NewGroup
    #y = sensors.Unitmotion
    y = sensors.Pose
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    start = time.time()
    DTC = DecisionTreeClassifier(criterion="entropy")
    DTC = DTC.fit(x_train,y_train)
    end = time.time()
    y_predDTC = DTC.predict(x_test)
    print((end-start))
    a = round((metrics.accuracy_score(y_test, y_predDTC)*100),2)
    print("DT: ",a)
    # labels = np.unique(y_test)
    # a = pd.DataFrame(metrics.confusion_matrix(y_test,y_predDTC, labels=labels), index=labels, columns=labels)
    # a.to_csv('machine/overlap80_2s/newgroup317/DT.csv',index=False)





    # start = time.time()
    # knn5 = KNeighborsClassifier(n_neighbors=3)
    # knn5.fit(x_train,y_train)
    # end = time.time()
    # y_predknn5 = knn5.predict(x_test)
    # print(end-start)
    # print("KNN 3 :",metrics.accuracy_score(y_test, y_predknn5))
    #
    #
    # start = time.time()
    # clf = ExtraTreesClassifier(n_estimators=10, random_state=0)
    # clf.fit(x_train,y_train)
    # end = time.time()
    # yrfc = clf.predict(x_test)
    # print(end-start)
    # print("ETC: ", metrics.accuracy_score(y_test, yrfc))
    #
    # start = time.time()
    # clf = RandomForestClassifier(n_estimators=10, random_state=0)
    # clf.fit(x_train,y_train)
    # end = time.time()
    # yrfc = clf.predict(x_test)
    # print(end-start)
    # print("RFC: ", metrics.accuracy_score(y_test, yrfc))



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