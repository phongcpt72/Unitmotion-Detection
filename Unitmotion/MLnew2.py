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
import pandas as pd
import warnings
warnings.simplefilter('ignore', FutureWarning)

def DTCresult(x_train, x_test, y_train, y_test):
    start = time.time()
    DTC = DecisionTreeClassifier(criterion="entropy")
    DTC = DTC.fit(x_train, y_train)
    end = time.time()
    y_predDTC = DTC.predict(x_test)
    accu = round((metrics.accuracy_score(y_test, y_predDTC)*100),2)
    pre  = round((precision_score(y_test, y_predDTC, average="macro")*100),2)
    f1 = round((f1_score(y_test, y_predDTC, average="macro")*100),2)
    recal = round((recall_score(y_test, y_predDTC, average="macro")*100),2)
    runtime = end-start

    return runtime, accu, pre, f1, recal

def ETCresult(x_train, x_test, y_train, y_test, ntrees):
    start = time.time()
    etc = ExtraTreesClassifier(n_estimators=ntrees, random_state=0)
    etc = etc.fit(x_train, y_train)
    end = time.time()
    yETC = etc.predict(x_test)
    accu = round((metrics.accuracy_score(y_test, yETC)*100),2)
    pre = round((precision_score(y_test, yETC, average="macro")*100),2)
    f1 = round((f1_score(y_test, yETC, average="macro")*100),2)
    recal = round((recall_score(y_test, yETC, average="macro")*100),2)
    runtime = end - start

    return runtime, accu, pre, f1, recal

def RFCresult(x_train, x_test, y_train, y_test, ntrees):
    start = time.time()
    rfc = RandomForestClassifier(n_estimators=ntrees, random_state=0)
    rfc = rfc.fit(x_train, y_train)
    end = time.time()
    yETC = rfc.predict(x_test)
    accu = round((metrics.accuracy_score(y_test, yETC)*100),2)
    pre = round((precision_score(y_test, yETC, average="macro")*100),2)
    f1 = round((f1_score(y_test, yETC, average="macro")*100),2)
    recal = round((recall_score(y_test, yETC, average="macro")*100),2)
    runtime = end - start

    return runtime, accu, pre, f1, recal

def KNNresult(x_train, x_test, y_train, y_test, k):
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn = knn.fit(x_train, y_train)
    end = time.time()
    y_predknn = knn.predict(x_test)
    accu = round((metrics.accuracy_score(y_test, y_predknn)*100),2)
    pre = round((precision_score(y_test, y_predknn, average="macro")*100),2)
    f1 = round((f1_score(y_test, y_predknn, average="macro")*100),2)
    recal = round((recall_score(y_test, y_predknn, average="macro")*100),2)
    runtime = end - start

    return runtime, accu, pre, f1, recal



if __name__ == '__main__':
    olarr = ['overlap60_1s', 'overlap40_1s', 'overlap50_2s', 'overlap80_2s']
    for i in range(len(olarr)):
        filecur = olarr[i]

        filename = 'HoldingStyles'
        name = 'Holding Styles'

        sensors = pd.read_csv(f'machine/{filecur}/newgroup317/{filename}.csv')
        #filename = 'StaticDynamic'
        savefile = (f'machine/{filecur}/newgroup317/{filename}ml.csv')

        tmpname = []
        tmpfeature = []
        tmptype = []
        tmpml = []
        tmpacc = []
        tmprecall = []
        tmppre = []
        tmpf1 = []
        tmpmltime = []
        tmpotime = []
        tmpet = []

        fnamearr = ['Max', 'Min', 'Mean', 'StandardDeviation']
        for j in range(len(fnamearr)):
            fname = fnamearr[j]
            # NewGroup
            feature_cols = [
                f'{fname}Ax', f'{fname}Ay', f'{fname}Az',
                # f'{fname}Gx', f'{fname}Gy', f'{fname}Gz',
                # f'{fname}Mx', f'{fname}My', f'{fname}Mz'
            ]

            x = sensors[feature_cols]
            #y = sensors.NewGroup
            #y = sensors.Unitmotion
            y = sensors.Pose
            if filecur == 'overlap80_2s':
                notype = 802
                otime = 0.4
            elif filecur == 'overlap50_2s':
                notype = 502
                otime = 1
            elif filecur == 'overlap60_1s':
                notype = 601
                otime = 0.4
            else:
                notype = 401
                otime = 0.6


            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

            runtime, accu, pre, f1, recal = DTCresult(x_train, x_test, y_train, y_test)
            tmpname.append(name)
            tmpfeature.append(fname)
            tmptype.append(notype)
            tmpml.append('DT')
            tmpacc.append(accu)
            tmprecall.append(recal)
            tmppre.append(pre)
            tmpf1.append(f1)
            tmpmltime.append(runtime)
            tmpotime.append(otime)
            tmpet.append(round(runtime+otime,2))

            knn = 3
            runtime, accu, pre, f1, recal = KNNresult(x_train, x_test, y_train, y_test, knn)
            tmpname.append(name)
            tmpfeature.append(fname)
            tmptype.append(notype)
            tmpml.append('KNN3')
            tmpacc.append(accu)
            tmprecall.append(recal)
            tmppre.append(pre)
            tmpf1.append(f1)
            tmpmltime.append(runtime)
            tmpotime.append(otime)
            tmpet.append(round(runtime+otime,2))

            knn = 7
            runtime, accu, pre, f1, recal = KNNresult(x_train, x_test, y_train, y_test, knn)
            tmpname.append(name)
            tmpfeature.append(fname)
            tmptype.append(notype)
            tmpml.append('KNN7')
            tmpacc.append(accu)
            tmprecall.append(recal)
            tmppre.append(pre)
            tmpf1.append(f1)
            tmpmltime.append(runtime)
            tmpotime.append(otime)
            tmpet.append(round(runtime+otime,2))

            ntrees = 100
            runtime, accu, pre, f1, recal = ETCresult(x_train, x_test, y_train, y_test, ntrees)
            tmpname.append(name)
            tmpfeature.append(fname)
            tmptype.append(notype)
            tmpml.append('ETC100')
            tmpacc.append(accu)
            tmprecall.append(recal)
            tmppre.append(pre)
            tmpf1.append(f1)
            tmpmltime.append(runtime)
            tmpotime.append(otime)
            tmpet.append(round(runtime+otime,2))

            ntrees = 50
            runtime, accu, pre, f1, recal = ETCresult(x_train, x_test, y_train, y_test, ntrees)
            tmpname.append(name)
            tmpfeature.append(fname)
            tmptype.append(notype)
            tmpml.append('ETC50')
            tmpacc.append(accu)
            tmprecall.append(recal)
            tmppre.append(pre)
            tmpf1.append(f1)
            tmpmltime.append(runtime)
            tmpotime.append(otime)
            tmpet.append(round(runtime+otime,2))

            ntrees = 25
            runtime, accu, pre, f1, recal = ETCresult(x_train, x_test, y_train, y_test, ntrees)
            tmpname.append(name)
            tmptype.append(notype)
            tmpfeature.append(fname)
            tmpml.append('ETC25')
            tmpacc.append(accu)
            tmprecall.append(recal)
            tmppre.append(pre)
            tmpf1.append(f1)
            tmpmltime.append(runtime)
            tmpotime.append(otime)
            tmpet.append(round(runtime+otime,2))

            ntrees = 10
            runtime, accu, pre, f1, recal = ETCresult(x_train, x_test, y_train, y_test, ntrees)
            tmpname.append(name)
            tmptype.append(notype)
            tmpfeature.append(fname)
            tmpml.append('ETC10')
            tmpacc.append(accu)
            tmprecall.append(recal)
            tmppre.append(pre)
            tmpf1.append(f1)
            tmpmltime.append(runtime)
            tmpotime.append(otime)
            tmpet.append(round(runtime+otime,2))

            ntrees = 100
            runtime, accu, pre, f1, recal = RFCresult(x_train, x_test, y_train, y_test, ntrees)
            tmpname.append(name)
            tmptype.append(notype)
            tmpml.append('RFC100')
            tmpfeature.append(fname)
            tmpacc.append(accu)
            tmprecall.append(recal)
            tmppre.append(pre)
            tmpf1.append(f1)
            tmpmltime.append(runtime)
            tmpotime.append(otime)
            tmpet.append(round(runtime+otime,2))

            ntrees = 50
            runtime, accu, pre, f1, recal = RFCresult(x_train, x_test, y_train, y_test, ntrees)
            tmpname.append(name)
            tmptype.append(notype)
            tmpfeature.append(fname)
            tmpml.append(f'RFC{ntrees}')
            tmpacc.append(accu)
            tmprecall.append(recal)
            tmppre.append(pre)
            tmpf1.append(f1)
            tmpmltime.append(runtime)
            tmpotime.append(otime)
            tmpet.append(round(runtime+otime,2))

            ntrees = 25
            runtime, accu, pre, f1, recal = RFCresult(x_train, x_test, y_train, y_test, ntrees)
            tmpname.append(name)
            tmptype.append(notype)
            tmpfeature.append(fname)
            tmpml.append(f'RFC{ntrees}')
            tmpacc.append(accu)
            tmprecall.append(recal)
            tmppre.append(pre)
            tmpf1.append(f1)
            tmpmltime.append(runtime)
            tmpotime.append(otime)
            tmpet.append(round(runtime+otime,2))

            ntrees = 10
            runtime, accu, pre, f1, recal = RFCresult(x_train, x_test, y_train, y_test, ntrees)
            tmpname.append(name)
            tmptype.append(notype)
            tmpfeature.append(fname)
            tmpml.append('RFC10')
            tmpacc.append(accu)
            tmprecall.append(recal)
            tmppre.append(pre)
            tmpf1.append(f1)
            tmpmltime.append(runtime)
            tmpotime.append(otime)
            tmpet.append(round(runtime+otime,2))

            result = pd.DataFrame({"Name": tmpname, "Feature": tmpfeature, "Type": tmptype, "ML": tmpml,"Accuracy": tmpacc,
                                   "Recall": tmprecall,"Precision" : tmppre,"F1": tmpf1,"MLTime": tmpmltime,
                                   "Otime": tmpotime, "Estimate Time": tmpet})
            result.to_csv(savefile, index=False)
            print('Done')






