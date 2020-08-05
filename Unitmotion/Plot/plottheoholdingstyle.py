from numpy import where
from collections import Counter
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import numpy as np

def RMS(tmp):
    return np.sqrt((np.square(tmp[0]))+(np.square(tmp[1]))+(np.square(tmp[2])))

def CalRMS(X,Y,Z):
    arrtmp = []
    for i in range(len(X)):
        a = X[i]
        b = Y[i]
        c = Z[i]
        tmp = [a,b,c]
        arrtmp.append(RMS(tmp))

    return arrtmp

def plot2D(Xa, Ya, y, strA, strB):
    aa = pd.DataFrame({"MaxAx": Xa, "MinAy": Ya})  # x,y in 2D
    X = np.array(aa)

    # summarize dataset shape
    print(X.shape, y.shape)
    # summarize observations by class label
    counter = Counter(y)
    print(counter)
    # summarize first few examples
    # for i in range(10):
    # 	print(X[i], y[i])
    # plot the dataset and color the by class label
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))

    plt.xlabel(strA)
    plt.ylabel(strB)
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.title('Texting')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    # define dataset
    va = read_csv('holdingstyle/Calling.csv')
    Type = va.GroupTypeName

    MaxAx = va.MaxAx
    MaxAy = va.MaxAy
    MaxAz = va.MaxAz

    MaxGx = va.MaxGx
    MaxGy = va.MaxGy
    MaxGz = va.MaxGz

    MaxMx = va.MaxMx
    MaxMy = va.MaxMy
    MaxMz = va.MaxMz

    MaxLiAx = va.MaxLiAx
    MaxLiAy = va.MaxLiAy
    MaxLiAz = va.MaxLiAz

    MaxP = va.MaxP

    MinAx = va.MinAx
    MinAy = va.MinAy
    MinAz = va.MinAz

    MinGx = va.MinGx
    MinGy = va.MinGy
    MinGz = va.MinGz

    MinMx = va.MinMx
    MinMy = va.MinMy
    MinMz = va.MinMz

    MinLiAx = va.MinLiAx
    MinLiAy = va.MinLiAy
    MinLiAz = va.MinLiAz

    MinP = va.MinP

    MeAx = va.MeanAx
    MeAy = va.MeanAy
    MeAz = va.MeanAz

    MeGx = va.MeanGx
    MeGy = va.MeanGy
    MeGz = va.MeanGz

    MeMx = va.MeanMx
    MeMy = va.MeanMy
    MeMz = va.MeanMz

    MeLiAx = va.MeanLiAx
    MeLiAy = va.MeanLiAy
    MeLiAz = va.MeanLiAz

    MeP = va.MeanP

    MedAx = va.MedianAx
    MedAy = va.MedianAy
    MedAz = va.MedianAz

    MedGx = va.MedianGx
    MedGy = va.MedianGy
    MedGz = va.MedianGz

    MedMx = va.MedianMx
    MedMy = va.MedianMy
    MedMz = va.MedianMz

    MedLiAx = va.MedianLiAx
    MedLiAy = va.MedianLiAy
    MedLiAz = va.MedianLiAz

    MedP = va.MedianP

    VarAx = va.VarianceAx
    VarAy = va.VarianceAy
    VarAz = va.VarianceAz

    VarGx = va.VarianceGx
    VarGy = va.VarianceGy
    VarGz = va.VarianceGz

    VarMx = va.VarianceMx
    VarMy = va.VarianceMy
    VarMz = va.VarianceMz

    VarLiAx = va.VarianceLiAx
    VarLiAy = va.VarianceLiAy
    VarLiAz = va.VarianceLiAz

    VarP = va.VarianceP

    StandardDeviationAx = va.StandardDeviationAx
    StandardDeviationAy = va.StandardDeviationAy
    StandardDeviationAz = va.StandardDeviationAz

    StandardDeviationGx = va.StandardDeviationGx
    StandardDeviationGy = va.StandardDeviationGy
    StandardDeviationGz = va.StandardDeviationGz

    StandardDeviationMx = va.StandardDeviationMx
    StandardDeviationMy = va.StandardDeviationMy
    StandardDeviationMz = va.StandardDeviationMz

    StandardDeviationLiAx = va.StandardDeviationLiAx
    StandardDeviationLiAy = va.StandardDeviationLiAy
    StandardDeviationLiAz = va.StandardDeviationLiAz

    StandardDeviationP = va.StandardDeviationP

    KurtosisAx = va.KurtosisAx
    KurtosisAy = va.KurtosisAy
    KurtosisAz = va.KurtosisAz

    KurtosisGx = va.KurtosisGx
    KurtosisGy = va.KurtosisGy
    KurtosisGz = va.KurtosisGz

    KurtosisMx = va.KurtosisMx
    KurtosisMy = va.KurtosisMy
    KurtosisMz = va.KurtosisMz

    KurtosisLiAx = va.KurtosisLiAx
    KurtosisLiAy = va.KurtosisLiAy
    KurtosisLiAz = va.KurtosisLiAz

    KurtosisP = va.KurtosisP

    SkeAx = va.SkeAx
    SkeAy = va.SkeAy
    SkeAz = va.SkeAz

    SkeGx = va.SkeGx
    SkeGy = va.SkeGy
    SkeGz = va.SkeGz

    SkeMx = va.SkeMx
    SkeMy = va.SkeMy
    SkeMz = va.SkeMz

    SkeLiAx = va.SkeLiAx
    SkeLiAy = va.SkeLiAy
    SkeLiAz = va.SkeLiAz

    SkeP = va.SkeP

    rmsmaxacc = CalRMS(MaxAx, MaxAy, MaxAz)
    rmsmaxgyr = CalRMS(MaxGx, MaxGy, MaxGz)
    rmsmaxmag = CalRMS(MaxMx, MaxMy, MaxMz)
    rmsmaxliacc = CalRMS(MaxLiAx, MaxLiAy, MaxLiAz)

    rmsminacc = CalRMS(MinAx, MinAy, MinAz)
    rmsmingyr = CalRMS(MinGx, MinGy, MinGz)
    rmsminmag = CalRMS(MinMx, MinMy, MinMz)
    rmsminliacc = CalRMS(MinLiAx, MinLiAy, MinLiAz)

    rmsmeanacc = CalRMS(MeAx, MeAy, MeAz)
    rmsmeangyr = CalRMS(MeGx, MeGy, MeGz)
    rmsmeanmag = CalRMS(MeMx, MeMy, MeMz)
    rmsmeanliacc = CalRMS(MeLiAx, MeLiAy, MeLiAz)

    rmsmedacc = CalRMS(MedAx, MedAy, MedAz)
    rmsmedgyr = CalRMS(MedGx, MedGy, MedGz)
    rmsmedmag = CalRMS(MedMx, MedMy, MedMz)
    rmsmedliacc = CalRMS(MedLiAx, MedLiAy, MedLiAz)

    rmsvaracc = CalRMS(VarAx, VarAy, VarAz)
    rmsvargyr = CalRMS(VarGx, VarGy, VarGz)
    rmsvarmag = CalRMS(VarMx, VarMy, VarMz)
    rmsvarliacc = CalRMS(VarLiAx, VarLiAy, VarLiAz)

    rmsstdacc = CalRMS(StandardDeviationAx, StandardDeviationAy, StandardDeviationAz)
    rmsstdgyr = CalRMS(StandardDeviationGx, StandardDeviationGy, StandardDeviationGz)
    rmsstdmag = CalRMS(StandardDeviationMx, StandardDeviationMy, StandardDeviationMz)
    rmsstdliacc = CalRMS(StandardDeviationLiAx, StandardDeviationLiAy, StandardDeviationLiAz)

    rmskuracc = CalRMS(KurtosisAx, KurtosisAy, KurtosisAz)
    rmskurgyr = CalRMS(KurtosisGx, KurtosisGy, KurtosisGz)
    rmskurmag = CalRMS(KurtosisMx, KurtosisMy, KurtosisMz)
    rmskurliacc = CalRMS(KurtosisLiAx, KurtosisLiAy, KurtosisLiAz)

    rmsskeacc = CalRMS(SkeAx, SkeAy, SkeAz)
    rmsskegyr = CalRMS(SkeGx, SkeGy, SkeGz)
    rmsskemag = CalRMS(SkeMx, SkeMy, SkeMz)
    rmsskeliacc = CalRMS(SkeLiAx, SkeLiAy, SkeLiAz)

    plot2D(rmsmaxacc, rmsmaxgyr, Type,'rmsmaxacc','rmsmaxgyr')
    plot2D(rmsmaxmag, rmsmaxliacc, Type, 'rmsmaxmag', 'rmsmaxliacc')
    plot2D(rmsminacc, rmsmingyr, Type, 'rmsminacc', 'rmsmingyr')
    plot2D(rmsminmag, rmsminliacc, Type, 'rmsminmag', 'rmsminliacc')

    plot2D(rmsmeanacc, rmsmeangyr, Type, 'rmsmeanacc', 'rmsmeangyr')
    plot2D(rmsmeanmag, rmsmeanliacc, Type, 'rmsmeanmag', 'rmsmeanliacc')
    plot2D(rmsmedacc, rmsmedgyr, Type, 'rmsmedacc', 'rmsmedgyr')
    plot2D(rmsmedmag, rmsmedliacc, Type, 'rmsmedmag', 'rmsmedliacc')

    plot2D(rmsvaracc, rmsvargyr, Type, 'rmsvaracc', 'rmsvargyr')
    plot2D(rmsvarmag, rmsvarliacc, Type, 'rmsvarmag', 'rmsvarliacc')
    plot2D(rmsstdacc, rmsstdgyr, Type, 'rmsstdacc', 'rmsstdgyr')
    plot2D(rmsstdmag, rmsstdliacc, Type, 'rmsstdmag', 'rmsstdliacc')

    plot2D(MaxAx, MaxAy, Type, 'MaxAx', 'MaxAy')
    plot2D(MaxAz, MaxGx, Type, 'MaxAz', 'MaxGx')
    plot2D(MaxGy, MaxGz, Type, 'MaxGy', 'MaxGz')

    plot2D(MaxMx, MaxMy, Type, 'MaxMx', 'MaxMy')
    plot2D(MaxMz, MaxLiAx, Type, 'MaxMz', 'MaxGx')
    plot2D(MaxLiAy, MaxLiAz, Type, 'MaxLiAy', 'MaxLiAz')

    plot2D(MinAx, MinAy, Type, 'MinAx', 'MinAy')
    plot2D(MinAz, MinGx, Type, 'MinAz', 'MinGx')
    plot2D(MinGy, MinGz, Type, 'MinGy', 'MinGz')

    plot2D(MinMx, MinMy, Type, 'MinMx', 'MinMy')
    plot2D(MinMz, MinLiAx, Type, 'MinMz', 'MinLiAx')
    plot2D(MinLiAy, MinLiAz, Type, 'MinLiAy', 'MinLiAz')

    plot2D(MeAx, MeAy, Type, 'MeAx', 'MeAy')
    plot2D(MeAz, MeGx, Type, 'MeAz', 'MeGx')
    plot2D(MeGy, MeGz, Type, 'MeGy', 'MeGz')

    plot2D(MeMx, MeMy, Type, 'MeMx', 'MeMy')
    plot2D(MeMz, MeLiAx, Type, 'MeMz', 'MeLiAx')
    plot2D(MeLiAy, MeLiAz, Type, 'MeLiAy', 'MeLiAz')

    plot2D(MedAx, MedAy, Type, 'MedAx', 'MedAy')
    plot2D(MedAz, MedGx, Type, 'MedAz', 'MedGx')
    plot2D(MedGy, MedGz, Type, 'MedGy', 'MedGz')

    plot2D(MedMx, MedMy, Type, 'MedMx', 'MedMy')
    plot2D(MedMz, MedLiAx, Type, 'MedMz', 'MedLiAx')
    plot2D(MedLiAy, MedLiAz, Type, 'MedLiAy', 'MedLiAz')

    plot2D(VarAx, MedMy, Type, 'VarAx', 'MedMy')
    plot2D(MedMz, MedLiAx, Type, 'MedMz', 'MedLiAx')
    plot2D(MedLiAy, MedLiAz, Type, 'MedLiAy', 'MedLiAz')

    plot2D(MedMx, VarAy, Type, 'MedMx', 'VarAy')
    plot2D(VarAz, VarGx, Type, 'VarAz', 'VarGx')
    plot2D(VarGy, VarGz, Type, 'VarGy', 'VarGz')

    plot2D(VarMx, VarMy, Type, 'VarMx', 'VarMy')
    plot2D(VarMz, VarLiAx, Type, 'VarMz', 'VarLiAx')
    plot2D(VarLiAy, VarLiAz, Type, 'VarLiAy', 'VarLiAz')

    plot2D(StandardDeviationAx, StandardDeviationAy, Type, 'StandardDeviationAx', 'StandardDeviationAy')
    plot2D(StandardDeviationAz, StandardDeviationGx, Type, 'StandardDeviationAz', 'StandardDeviationGx')
    plot2D(StandardDeviationGy, StandardDeviationGz, Type, 'StandardDeviationGy', 'StandardDeviationGz')

    plot2D(StandardDeviationMx, StandardDeviationMy, Type, 'StandardDeviationMx', 'StandardDeviationMy')
    plot2D(StandardDeviationMz, StandardDeviationLiAx, Type, 'StandardDeviationMz', 'StandardDeviationLiAx')
    plot2D(StandardDeviationLiAy, StandardDeviationLiAz, Type, 'StandardDeviationLiAy', 'StandardDeviationLiAz')