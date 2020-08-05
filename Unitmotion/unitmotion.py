import pandas as pd
import numpy as np
import statistics as sta
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import iqr
def overlapfunction(N, arr):
    overlapnoas = int(N*0.6)
    tmp = []
    for i in range(overlapnoas):
        tmp.append(i)
    tmp.sort(reverse=True)
    for j in range(len(tmp)):
        del arr[tmp[j]]
    return arr

""" Calculate Max, Min"""
def CalMaxMin(arr,windowtime):
    tmp = []
    i = 0
    maxarr = []
    minarr = []
    countbreak = len(arr)
    while i < countbreak:
        tmp.append(arr[i])
        if len(tmp) == windowtime:
            maxarr.append(np.max(tmp))
            minarr.append(np.min(tmp))
            tmp = overlapfunction(windowtime,tmp)
            i+=1
        else:
            i += 1
            if i == countbreak:
                maxarr.append(np.max(tmp))
                minarr.append(np.min(tmp))
                break
    return maxarr,minarr

"""Find type following ID"""
def Types(arr,windowtime):
    tmp = []
    i = 0
    typearr = []
    countbreak = len(arr)
    while i < countbreak:
        tmp.append(arr[i])
        if len(tmp) == windowtime:
            typearr.append(np.max(tmp))
            tmp = overlapfunction(windowtime, tmp)
            i += 1
        else:
            i += 1
            if i == countbreak:
                typearr.append(np.max(tmp))
                break
    return typearr

def CalStandardDeviation(arr,windowtime):
    tmp = []
    i = 0
    StandardDeviationarr = []
    countbreak = len(arr)
    while i < countbreak:
        tmp.append(arr[i])
        if len(tmp) == windowtime:
            StandardDeviationarr.append(sta.stdev(tmp))
            tmp = overlapfunction(windowtime, tmp)
            i += 1
        else:
            i += 1
            if i == countbreak:
                StandardDeviationarr.append(sta.stdev(tmp))
                break
    return StandardDeviationarr

""" Calculate Mean"""
def CalMean(arr,windowtime):
    tmp = []
    i = 0
    meanarr = []
    countbreak = len(arr)
    while i < countbreak:
        tmp.append(arr[i])
        if len(tmp) == windowtime:
            meanarr.append(sta.mean(tmp))
            tmp = overlapfunction(windowtime, tmp)
            i += 1
        else:
            i += 1
            if i == countbreak:
                meanarr.append(sta.mean(tmp))
                break
    return meanarr

def CalKurtosisSkew(arr,windowtime):
    tmp = []
    i = 0
    Kurtosisarr = []
    Skewarr = []
    countbreak = len(arr)
    while i < countbreak:
        tmp.append(arr[i])
        if len(tmp) == windowtime:
            Kurtosisarr.append(kurtosis(tmp))
            Skewarr.append(skew(tmp))
            tmp = overlapfunction(windowtime, tmp)
            i += 1
        else:
            i += 1
            if i == countbreak:
                Kurtosisarr.append(kurtosis(tmp))
                Skewarr.append(skew(tmp))
                break
    return Kurtosisarr,Skewarr

def CalVariance(arr,windowtime):
    tmp = []
    i = 0
    variancearr = []
    countbreak = len(arr)
    while i < countbreak:
        tmp.append(arr[i])
        if len(tmp) == windowtime:
            variancearr.append(sta.variance(tmp))
            tmp = overlapfunction(windowtime, tmp)
            i += 1
        else:
            i += 1
            if i == countbreak:
                variancearr.append(sta.variance(tmp))
                break
    return variancearr

def SMA(x,y,z):
    sum = 0
    for i in range(len(x)):
        sum += abs(x[i]) + abs(y[i]) + abs(z[i])
    return sum/len(x)

def CalSMA(x,y,z,windowtime):
    xarr = []
    yarr = []
    zarr = []
    smaarr = []
    countbreak = len(x)
    i = 0
    while i < countbreak:
        xarr.append(x[i])
        yarr.append(y[i])
        zarr.append(z[i])
        if len(xarr) == windowtime:
            smaarr.append(SMA(xarr,yarr,zarr))
            xarr = overlapfunction(windowtime, xarr)
            yarr = overlapfunction(windowtime, yarr)
            zarr = overlapfunction(windowtime, zarr)
            i+=1
        else:
            i += 1
            if i == countbreak:
                smaarr.append(SMA(xarr,yarr,zarr))
                break
    return smaarr

def CalMedian(arr,windowtime):
    tmp = []
    i = 0
    medianarr = []
    countbreak = len(arr)
    while i < countbreak:
        tmp.append(arr[i])
        if len(tmp) == windowtime:
            medianarr.append(sta.median(tmp))
            tmp = overlapfunction(windowtime, tmp)
            i += 1
        else:
            i += 1
            if i == countbreak:
                medianarr.append(sta.median(tmp))
                break
    return medianarr

def calLPF(alpha, value, tmparr, info):
    if len(tmparr) == 0:
        va = alpha * value;
        tmparr.append(va)
    else:
        va = alpha * value + (1-alpha) * tmparr[info-1]
        tmparr.append(va)
    return va

def calalpha(cutoff, fs):
    dt = 1/fs
    T  = 1/cutoff
    return round(dt/(T+dt),2)

def LPF(alpha,rawarr):
    i = 0
    tmparr = []
    lpfarr = []
    while i < len(rawarr):
        va = calLPF(alpha,rawarr[i],tmparr, i)
        i+=1
        lpfarr.append(va)
    return lpfarr

def LPFOrder(file,alpha):
    Ax = file.Ax
    Ay = file.Ay
    Az = file.Az
    Gx = file.Gx
    Gy = file.Gy
    Gz = file.Gz
    Mx = file.Mx
    My = file.My
    Mz = file.Mz
    LiAx = file.LiAx
    LiAy = file.LiAy
    LiAz = file.LiAz
    P = file.P

    ax = LPF(alpha, Ax)
    ax = LPF(alpha, ax)
    ay = LPF(alpha, Ay)
    ay = LPF(alpha, ay)
    az = LPF(alpha, Az)
    az = LPF(alpha, az)

    gx = LPF(alpha, Gx)
    gx = LPF(alpha, gx)
    gy = LPF(alpha, Gy)
    gy = LPF(alpha, gy)
    gz = LPF(alpha, Gz)
    gz = LPF(alpha, gz)

    mx = LPF(alpha, Mx)
    mx = LPF(alpha, mx)
    my = LPF(alpha, My)
    my = LPF(alpha, my)
    mz = LPF(alpha, Mz)
    mz = LPF(alpha, mz)

    liax = LPF(alpha, LiAx)
    liax = LPF(alpha, liax)
    liay = LPF(alpha, LiAy)
    liay = LPF(alpha, liay)
    liaz = LPF(alpha, LiAz)
    liaz = LPF(alpha, liaz)

    p = LPF(alpha, P)
    p = LPF(alpha, p)

    return ax, ay, az, gx, gy, gz, mx, my, mz, liax, liay, liaz, p

def Mag(x,y,z):
    return np.sqrt((np.square(x))+(np.square(y))+(np.square(z)))

def CalMag(x,y,z,windowtime):
    xarr = []
    yarr = []
    zarr = []
    magarr = []
    countbreak = len(x)
    i = 0
    while i < countbreak:
        xarr.append(x[i])
        yarr.append(y[i])
        zarr.append(z[i])
        if len(xarr) == windowtime:
            magarr.append(Mag(xarr,yarr,zarr))
            xarr = overlapfunction(windowtime, xarr)
            yarr = overlapfunction(windowtime, yarr)
            zarr = overlapfunction(windowtime, zarr)
            i+=1
        else:
            i += 1
            if i == countbreak:
                magarr.append(Mag(xarr,yarr,zarr))
                break
    return magarr

def CalInterquartileRange(arr,windowtime):
    tmp = []
    i = 0
    iqrarr = []
    countbreak = len(arr)
    while i < countbreak:
        tmp.append(arr[i])
        if len(tmp) == windowtime:
            iqrarr.append(iqr(tmp, interpolation= 'midpoint'))
            tmp = overlapfunction(windowtime, tmp)
            i += 1
        else:
            i += 1
            if i == countbreak:
                iqrarr.append(iqr(tmp, interpolation= 'midpoint'))
                break
    return iqrarr

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

if __name__ == '__main__':

    posingname = ['Calling', 'Texting', 'Swinging', 'Pocket']
    for i in range(len(posingname)):
        namepose = posingname[i]
        filessss = 1
        while filessss < 11:
            pose = namepose
            file = pd.read_csv(f'addtype/{namepose}/Sensors_{filessss}.csv')
            resultname = f'machine/{namepose}/Sensors_{filessss}.csv'
            filessss += 1
            Value = file.Value
            NewType = file.Type
            GroupType = file.GroupType
            Activities = file.Activities

            N = 25
            #overlapno = 0.4  # % 0.2 = 80%, 0.5 = 50%
            windowtime = N * 1  # s
            fs = 25
            cutoff = 1.5
            alpha = calalpha(cutoff,fs)
            ax, ay, az, gx, gy, gz, mx, my, mz, liax, liay, liaz, p = LPFOrder(file, alpha)

            MaxAx, MinAx = CalMaxMin(ax, windowtime)
            MaxAy, MinAy = CalMaxMin(ay, windowtime)
            MaxAz, MinAz = CalMaxMin(az, windowtime)

            MaxGx, MinGx = CalMaxMin(gx, windowtime)
            MaxGy, MinGy = CalMaxMin(gy, windowtime)
            MaxGz, MinGz = CalMaxMin(gz, windowtime)

            MaxMx, MinMx = CalMaxMin(mx, windowtime)
            MaxMy, MinMy = CalMaxMin(my, windowtime)
            MaxMz, MinMz = CalMaxMin(mz, windowtime)

            MaxLiAx, MinLiAx = CalMaxMin(liax, windowtime)
            MaxLiAy, MinLiAy = CalMaxMin(liay, windowtime)
            MaxLiAz, MinLiAz = CalMaxMin(liaz, windowtime)

            MaxP, MinP = CalMaxMin(p, windowtime)

            MeanAx = CalMean(ax, windowtime)
            MeanAy = CalMean(ay, windowtime)
            MeanAz = CalMean(az, windowtime)

            MeanGx = CalMean(gx, windowtime)
            MeanGy = CalMean(gy, windowtime)
            MeanGz = CalMean(gz, windowtime)

            MeanMx = CalMean(mx, windowtime)
            MeanMy = CalMean(my, windowtime)
            MeanMz = CalMean(mz, windowtime)

            MeanLiAx = CalMean(liax, windowtime)
            MeanLiAy = CalMean(liay, windowtime)
            MeanLiAz = CalMean(liaz, windowtime)

            MeanP = CalMean(p, windowtime)

            MedianAx = CalMedian(ax, windowtime)
            MedianAy = CalMedian(ay, windowtime)
            MedianAz = CalMedian(az, windowtime)

            MedianGx = CalMedian(gx, windowtime)
            MedianGy = CalMedian(gy, windowtime)
            MedianGz = CalMedian(gz, windowtime)

            MedianMx = CalMedian(mx, windowtime)
            MedianMy = CalMedian(my, windowtime)
            MedianMz = CalMedian(mz, windowtime)

            MedianLiAx = CalMedian(liax, windowtime)
            MedianLiAy = CalMedian(liay, windowtime)
            MedianLiAz = CalMedian(liaz, windowtime)

            MedianP = CalMedian(p, windowtime)

            StandardDeviationAx = CalStandardDeviation(ax, windowtime)
            StandardDeviationAy = CalStandardDeviation(ay, windowtime)
            StandardDeviationAz = CalStandardDeviation(az, windowtime)

            StandardDeviationGx = CalStandardDeviation(gx, windowtime)
            StandardDeviationGy = CalStandardDeviation(gy, windowtime)
            StandardDeviationGz = CalStandardDeviation(gz, windowtime)

            StandardDeviationMx = CalStandardDeviation(mx, windowtime)
            StandardDeviationMy = CalStandardDeviation(my, windowtime)
            StandardDeviationMz = CalStandardDeviation(mz, windowtime)

            StandardDeviationLiAx = CalStandardDeviation(liax, windowtime)
            StandardDeviationLiAy = CalStandardDeviation(liay, windowtime)
            StandardDeviationLiAz = CalStandardDeviation(liaz, windowtime)

            StandardDeviationP = CalStandardDeviation(p, windowtime)

            VarianceAx = CalVariance(ax, windowtime)
            VarianceAy = CalVariance(ay, windowtime)
            VarianceAz = CalVariance(az, windowtime)

            VarianceGx = CalVariance(gx, windowtime)
            VarianceGy = CalVariance(gy, windowtime)
            VarianceGz = CalVariance(gz, windowtime)

            VarianceMx = CalVariance(mx, windowtime)
            VarianceMy = CalVariance(my, windowtime)
            VarianceMz = CalVariance(mz, windowtime)

            VarianceLiAx = CalVariance(liax, windowtime)
            VarianceLiAy = CalVariance(liay, windowtime)
            VarianceLiAz = CalVariance(liaz, windowtime)

            VarianceP = CalVariance(p, windowtime)

            KurtosisAx, SkeAx = CalKurtosisSkew(ax, windowtime)
            KurtosisAy, SkeAy = CalKurtosisSkew(ay, windowtime)
            KurtosisAz, SkeAz = CalKurtosisSkew(az, windowtime)

            KurtosisGx, SkeGx = CalKurtosisSkew(gx, windowtime)
            KurtosisGy, SkeGy = CalKurtosisSkew(gy, windowtime)
            KurtosisGz, SkeGz = CalKurtosisSkew(gz, windowtime)

            KurtosisMx, SkeMx = CalKurtosisSkew(mx, windowtime)
            KurtosisMy, SkeMy = CalKurtosisSkew(my, windowtime)
            KurtosisMz, SkeMz = CalKurtosisSkew(mz, windowtime)

            KurtosisLiAx, SkeLiAx = CalKurtosisSkew(liax, windowtime)
            KurtosisLiAy, SkeLiAy = CalKurtosisSkew(liay, windowtime)
            KurtosisLiAz, SkeLiAz = CalKurtosisSkew(liaz, windowtime)

            KurtosisP, SkeP = CalKurtosisSkew(p, windowtime)

            SMAAcc = CalSMA(ax, ay, az, windowtime)
            SMAGyr = CalSMA(gx, gy, gz, windowtime)
            SMAMag = CalSMA(mx, my, mz, windowtime)
            SMALiAcc = CalSMA(liax, liay, liaz, windowtime)

            MagAcc1 = CalRMS(ax, ay, az)
            MagGyr1 = CalRMS(gx, gy, gz)
            MagMag1 = CalRMS(mx, my, mz)
            MagLiAcc1 = CalRMS(liax, liay, liaz)

            MeanMagAcc = CalMean(MagAcc1, windowtime)
            MeanMagGyr = CalMean(MagGyr1, windowtime)
            MeanMagMag = CalMean(MagMag1, windowtime)
            MeanMagLiAcc = CalMean(MagLiAcc1, windowtime)

            STDMagAcc = CalStandardDeviation(MagAcc1,windowtime)
            STDMagGyr = CalStandardDeviation(MagGyr1, windowtime)
            STDMagMag = CalStandardDeviation(MagMag1, windowtime)
            STDMagLiAcc = CalStandardDeviation(MagLiAcc1, windowtime)

            VarMagAcc = CalVariance(MagAcc1, windowtime)
            VarMagGyr = CalVariance(MagGyr1, windowtime)
            VarMagMag = CalVariance(MagMag1, windowtime)
            VarMagLiAcc = CalVariance(MagLiAcc1, windowtime)

            IqrMagAcc = CalInterquartileRange(MagAcc1, windowtime)
            IqrMagGyr = CalInterquartileRange(MagGyr1, windowtime)
            IqrMagMag = CalInterquartileRange(MagMag1, windowtime)
            IqrMagLiAcc = CalInterquartileRange(MagLiAcc1, windowtime)

            NewGroupType = Types(GroupType, windowtime)
            NewType = Types(NewType,windowtime)
            Activities = Types(Activities,windowtime)

            idtmp = []
            sums = 0
            for i in range(len(NewGroupType)):
                idtmp.append(sums)
                sums = round(sums + 1, 2)

            arrpose = []
            for i in range(len(NewType)):
                arrpose.append(pose)

            unitmotion = []
            for i in range(len(NewGroupType)):
                if NewGroupType[i] == 0:
                    unitmotion.append('Stading')
                elif NewGroupType[i] == 1:
                    unitmotion.append('Phone on table')
                elif NewGroupType[i] == 2:
                    unitmotion.append('Grab the phone')
                elif NewGroupType[i] == 3:
                    unitmotion.append('Put the phone')
                elif NewGroupType[i] == 4:
                    unitmotion.append('Walking')
                elif NewGroupType[i] == 5:
                    unitmotion.append('Passing the door')
                elif NewGroupType[i] == 6:
                    unitmotion.append('Downstairs')
                elif NewGroupType[i] == 7:
                    unitmotion.append('Upstairs')



            result = pd.DataFrame({"Timestamp": idtmp,
                "MaxAx": MaxAx, "MaxAy": MaxAy, "MaxAz": MaxAz,
                "MaxGx": MaxGx, "MaxGy": MaxGy, "MaxGz": MaxGz,
                "MaxMx": MaxMx, "MaxMy": MaxMy, "MaxMz": MaxMz,
                "MaxLiAx": MaxLiAx, "MaxLiAy": MaxLiAy, "MaxLiAz": MaxLiAz, "MaxP": MaxP,
                "MinAx": MinAx, "MinAy": MinAy, "MinAz": MinAz,
                "MinGx": MinGx, "MinGy": MinGy, "MinGz": MinGz,
                "MinMx": MinMx, "MinMy": MinMy, "MinMz": MinMz,
                "MinLiAx": MinLiAx, "MinLiAy": MinLiAy, "MinLiAz": MinLiAz, "MinP": MinP,
                "MeanAx": MeanAx, "MeanAy": MeanAy, "MeanAz": MeanAz,
                "MeanGx": MeanGx, "MeanGy": MeanGy, "MeanGz": MeanGz,
                "MeanMx": MeanMx, "MeanMy": MeanMy, "MeanMz": MeanMz,
                "MeanLiAx": MeanLiAx, "MeanLiAy": MeanLiAy, "MeanLiAz": MeanLiAz, "MeanP": MeanP,
                "StandardDeviationAx": StandardDeviationAx, "StandardDeviationAy": StandardDeviationAy,
                "StandardDeviationAz": StandardDeviationAz,
                "StandardDeviationGx": StandardDeviationGx, "StandardDeviationGy": StandardDeviationGy,
                "StandardDeviationGz": StandardDeviationGz,
                "StandardDeviationMx": StandardDeviationMx, "StandardDeviationMy": StandardDeviationMy,
                "StandardDeviationMz": StandardDeviationMz,
                "StandardDeviationLiAx": StandardDeviationLiAx, "StandardDeviationLiAy": StandardDeviationLiAy,
                "StandardDeviationLiAz": StandardDeviationLiAz, "StandardDeviationP": StandardDeviationP,
                "MedianAx": MedianAx, "MedianAy": MedianAy, "MedianAz": MedianAz,
                "MedianGx": MedianGx, "MedianGy": MedianGy, "MedianGz": MedianGz,
                "MedianMx": MedianMx, "MedianMy": MedianMy, "MedianMz": MedianMz,
                "MedianLiAx": MedianLiAx, "MedianLiAy": MedianLiAy, "MedianLiAz": MedianLiAz, "MedianP": MedianP,
                "VarianceAx": VarianceAx, "VarianceAy": VarianceAy, "VarianceAz": VarianceAz,
                "VarianceGx": VarianceGx, "VarianceGy": VarianceGy, "VarianceGz": VarianceGz,
                "VarianceMx": VarianceMx, "VarianceMy": VarianceMy, "VarianceMz": VarianceMz,
                "VarianceLiAx": VarianceLiAx, "VarianceLiAy": VarianceLiAy, "VarianceLiAz": VarianceLiAz, "VarianceP": VarianceP,
                "KurtosisAx": KurtosisAx, "KurtosisAy": KurtosisAy, "KurtosisAz": KurtosisAz,
                "KurtosisGx": KurtosisGx, "KurtosisGy": KurtosisGy, "KurtosisGz": KurtosisGz,
                "KurtosisMx": KurtosisMx, "KurtosisMy": KurtosisMy, "KurtosisMz": KurtosisMz,
                "KurtosisLiAx": KurtosisLiAx, "KurtosisLiAy": KurtosisLiAy, "KurtosisLiAz": KurtosisLiAz, "KurtosisP": KurtosisP,
                "SkeAx": SkeAx, "SkeAy": SkeAy, "SkeAz": SkeAz,
                "SkeGx": SkeGx, "SkeGy": SkeGy, "SkeGz": SkeGz,
                "SkeMx": SkeMx, "SkeMy": SkeMy, "SkeMz": SkeMz,
                "SkeLiAx": SkeLiAx, "SkeLiAy": SkeLiAy, "SkeLiAz": SkeLiAz, "SkeP": SkeP,
                "SMAAcc": SMAAcc, "SMAGyr": SMAGyr, "SMAMag": SMAMag, "SMALiAcc": SMALiAcc,
                "MeanMagAcc": MeanMagAcc,"MeanMagGyr": MeanMagGyr,"MeanMagMag": MeanMagMag, "MeanMagLiAcc": MeanMagLiAcc,
                "IqrMagAcc": IqrMagAcc, "IqrMagGyr": IqrMagGyr, "IqrMagMag": IqrMagMag, "IqrMagLiAcc": IqrMagLiAcc,
                "STDMagAcc": STDMagAcc, "STDMagGyr": STDMagGyr, "STDMagMag": STDMagMag, "STDMagLiAcc": STDMagLiAcc,
                "VarMagAcc": VarMagAcc, "VarMagGyr": VarMagGyr, "VarMagMag": VarMagMag, "VarMagLiAcc": VarMagLiAcc,
                "GroupType": NewGroupType,"Type": NewType,"Pose": arrpose,"Unitmotion": unitmotion,"Activities":Activities})

            result.to_csv(resultname, index=False)
        print("Done")





# import numpy as np
# from scipy.stats import iqr
#
# data = [32, 36, 46, 47, 56, 69, 75, 79, 79, 88, 89, 91, 92, 93, 96, 97,
#         101, 105, 112, 116]
#
# # First quartile (Q1)
# Q1 = np.median(data[:10])
#
# # Third quartile (Q3)
# Q3 = np.median(data[10:])
#
# # Interquartile range (IQR)
# IQR = Q3 - Q1
#
# print(len(data))
# print(IQR)
# print(iqr(data, interpolation = 'midpoint'))