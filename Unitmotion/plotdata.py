import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal

def plotdata(time,a,b,c,d,e, namea, nameb, namec, named, savename):

    plt.figure(1)
    plt.suptitle(titlename +' - ' + savename, fontsize=14, fontweight='bold')
    plt.subplot(5, 1, 1)
    plt.plot(time, a, color='red', label=namea)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 2)
    plt.plot(time, b, color='blue', label=nameb)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 3)
    plt.plot(time, c, color='red', label=namec)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 4)
    plt.plot(time, d, color='blue', label=named)
    plt.legend()
    plt.grid(True)
    plt.subplot(5, 1, 5)
    plt.plot(time, e, color='green', label='Type')
    plt.ylim(0,6)
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time [sec]')
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    #plt.savefig('analysispose/'+titlename+'/'+savename+'.png', dpi=100)
    #plt.close()
    plt.show()

def plotdata1(time,a,b,c,d, namea, nameb, namec, savename):

    plt.figure(1)
    plt.subplot(4, 1, 1)
    plt.suptitle(titlename +' - ' + savename, fontsize=14, fontweight='bold')
    plt.plot(time, a, color='red', label=namea)
    plt.legend()
    plt.grid(True)
    plt.subplot(4, 1, 2)
    plt.plot(time, b, color='blue', label=nameb)
    plt.legend()
    plt.grid(True)
    plt.subplot(4, 1, 3)
    plt.plot(time, c, color='red', label=namec)
    plt.legend()
    plt.grid(True)
    plt.subplot(4, 1, 4)
    plt.plot(time, d, color='blue', label='Type')
    plt.ylim(0,6)
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time [sec]')
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    #plt.savefig('analysispose/'+titlename+'/'+savename+'.png', dpi=100)
    #plt.close()
    plt.show()

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

def processingtype(idarr,tmparr, typearr):
    callingarr = [None for _ in range(len(tmparr))]
    pocketarr = [None for _ in range(len(tmparr))]
    swingingarr = [None for _ in range(len(tmparr))]
    textingarr = [None for _ in range(len(tmparr))]

    callingidarr = [None for _ in range(len(idarr))]
    pocketidarr = [None for _ in range(len(idarr))]
    swingingidarr = [None for _ in range(len(idarr))]
    textingidarr = [None for _ in range(len(idarr))]

    for i in range(len(typearr)):
        if typearr[i] == 1:
            callingarr[i] = tmparr[i]
            callingidarr[i] = idarr[i]
        elif typearr[i] == 2:
            pocketarr[i] = tmparr[i]
            pocketidarr[i] = idarr[i]
        elif typearr[i] == 3:
            swingingarr[i] = tmparr[i]
            swingingidarr[i] = idarr[i]
        elif typearr[i] == 4:
            textingarr[i] = tmparr[i]
            textingidarr[i] = idarr[i]

    return callingarr,callingidarr,pocketarr,pocketidarr,swingingarr,swingingidarr,textingarr,textingidarr

if __name__ == '__main__':


    titlename = 'Texting'
    va = pd.read_csv('Plot/'+ titlename +'.csv')

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

    #Type = va.Type
    Type = va.GroupType
    ID = va.Timestamp
    idtmp = []
    sums = 0
    for i in range(len(Type)):
        idtmp.append(sums)
        sums = round(sums + 0.04, 2)

    ID = idtmp

    rmsmaxacc = CalRMS(MaxAx,MaxAy,MaxAz)
    rmsmaxgyr = CalRMS(MaxGx,MaxGy,MaxGz)
    rmsmaxmag = CalRMS(MaxMx,MaxMy,MaxMz)
    rmsmaxliacc = CalRMS(MaxLiAx,MaxLiAy,MaxLiAz)

    rmsminacc = CalRMS(MinAx, MinAy, MinAz)
    rmsmingyr = CalRMS(MinGx,MinGy,MinGz)
    rmsminmag = CalRMS(MinMx,MinMy,MinMz)
    rmsminliacc = CalRMS(MinLiAx,MinLiAy,MinLiAz)

    rmsmeanacc = CalRMS(MeAx,MeAy,MeAz)
    rmsmeangyr = CalRMS(MeGx,MeGy,MeGz)
    rmsmeanmag = CalRMS(MeMx,MeMy,MeMz)
    rmsmeanliacc = CalRMS(MeLiAx,MeLiAy,MeLiAz)

    rmsmedacc = CalRMS(MedAx, MedAy, MedAz)
    rmsmedgyr = CalRMS(MedGx, MedGy, MedGz)
    rmsmedmag = CalRMS(MedMx, MedMy, MedMz)
    rmsmedliacc = CalRMS(MedLiAx, MedLiAy, MedLiAz)

    rmsvaracc = CalRMS(VarAx, VarAy, VarAz)
    rmsvargyr = CalRMS(VarGx, VarGy, VarGz)
    rmsvarmag = CalRMS(VarMx, VarMy, VarMz)
    rmsvarliacc = CalRMS(VarLiAx, VarLiAy, VarLiAz)

    rmsstdacc = CalRMS(StandardDeviationAx,StandardDeviationAy,StandardDeviationAz)
    rmsstdgyr = CalRMS(StandardDeviationGx,StandardDeviationGy,StandardDeviationGz)
    rmsstdmag = CalRMS(StandardDeviationMx,StandardDeviationMy,StandardDeviationMz)
    rmsstdliacc = CalRMS(StandardDeviationLiAx,StandardDeviationLiAy,StandardDeviationLiAz)

    rmskuracc = CalRMS(KurtosisAx, KurtosisAy, KurtosisAz)
    rmskurgyr = CalRMS(KurtosisGx, KurtosisGy, KurtosisGz)
    rmskurmag = CalRMS(KurtosisMx, KurtosisMy, KurtosisMz)
    rmskurliacc = CalRMS(KurtosisLiAx, KurtosisLiAy, KurtosisLiAz)

    rmsskeacc = CalRMS(SkeAx, SkeAy, SkeAz)
    rmsskegyr = CalRMS(SkeGx, SkeGy, SkeGz)
    rmsskemag = CalRMS(SkeMx, SkeMy, SkeMz)
    rmsskeliacc = CalRMS(SkeLiAx, SkeLiAy, SkeLiAz)

    # import matplotlib.pyplot as plt
    # import matplotlib.colors as colors
    # import matplotlib.cm as cmx
    # from pandas import read_csv
    #
    # df = va
    #
    # df['rmsstdacc'] = rmsstdacc
    # pos = df.loc[:, ["PoseName"]].groupby("PoseName").count().reset_index()
    #
    # # create a new column in the dataframe which contains the numeric value
    # tag_to_index = lambda x: pos.loc[pos.PoseName == x.PoseName].index[0]
    # df.loc[:, "name_index"] = df.loc[:, ["PoseName"]].apply(tag_to_index, axis=1)
    #
    # # Set the color map to match the number of species
    # hot = plt.get_cmap('hot')
    # cNorm = colors.Normalize(vmin=0, vmax=len(pos))
    # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)
    #
    # # Get unique names of species
    # for (name, group) in df.groupby("name_index"):
    #     plt.scatter(group.Timestamp, group.rmsstdacc, s=15, label=pos.iloc[name].get("PoseName"),
    #                 color=scalarMap.to_rgba(name))
    #
    # plt.xlabel('Timestamp')
    # plt.ylabel('Magnitude Standard Deviation Accelerometer')
    # plt.title('Grab Phone')
    # plt.legend()
    # plt.show()









    #
    #
    # callingarr, callingidarr, pocketarr, pocketidarr, swingingarr, swingingidarr, textingarr, textingidarr = processingtype(ID,rmsmaxacc,Type)
    #
    # dfcalling = pd.DataFrame({"id":callingidarr,"callingarr":callingarr})
    # dfpocket = pd.DataFrame({"id": pocketidarr, "pocketarr": pocketarr})
    # dfswinging = pd.DataFrame({"id": swingingidarr, "swingingarr": swingingarr})
    # dftexting = pd.DataFrame({"id": textingidarr, "textingarr": textingarr})
    # plt.show()
    #
    # ax1 = dfcalling.plot(kind='scatter', x='id', y='callingarr', color='r')
    # ax2 = dfpocket.plot(kind='scatter', x='id', y='pocketarr', color='g', ax=ax1)
    # ax3 = dfswinging.plot(kind='scatter', x='id', y='swingingarr', color='b', ax=ax1)
    # ax4 = dftexting.plot(kind='scatter', x='id', y='textingarr', color='c', ax=ax1)
    # plt.show()






    # import hvplot.pandas
    # import hvplot
    #
    # df = pd.DataFrame(np.random.randn(100, 6), columns=['a', 'b', 'c', 'd', 'e', 'f'])
    #
    # df.hvplot(x='a', y=['b', 'c', 'd', 'e'], kind='scatter')
    # hvplot.show()


    # fs = 25
    # rmsmeanacc = np.asarray(rmsmeanacc)
    # f, t, Sxx = signal.spectrogram(rmsmeanacc, fs)
    # plt.pcolormesh(t, f, Sxx)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()




    plotdata(ID, rmsmaxacc, rmsmaxgyr, rmsmaxmag,rmsmaxliacc, Type,'rmsmaxacc','rmsmaxgyr','rmsmaxmag','rmsmaxliacc','RMSMax')
    plotdata(ID, rmsminacc, rmsmingyr, rmsminmag, rmsminliacc, Type, 'rmsminacc', 'rmsmingyr', 'rmsminmag', 'rmsminliacc','RMSMin')
    plotdata(ID, rmsmeanacc, rmsmeangyr, rmsmeanmag, rmsmeanliacc, Type, 'rmsmeanacc', 'rmsmeangyr', 'rmsmeanmag',
             'rmsmeanliacc','RMSMean')
    plotdata(ID, rmsmedacc, rmsmedgyr, rmsmedmag, rmsmedliacc, Type, 'rmsmedacc', 'rmsmedgyr', 'rmsmedmag',
             'rmsmedliacc','RMSMedian')
    plotdata(ID, rmsvaracc, rmsvargyr, rmsvarmag, rmsvarliacc, Type, 'rmsvaracc', 'rmsvargyr', 'rmsvarmag',
             'rmsvarliacc','RMSVar')
    plotdata(ID, rmsstdacc, rmsstdgyr, rmsstdmag, rmsstdliacc, Type, 'rmsstdacc', 'rmsstdgyr', 'rmsstdmag',
             'rmsstdliacc','RMSSTD')
    plotdata(ID, rmskuracc, rmskurgyr, rmskurmag, rmskurliacc, Type, 'rmskuracc', 'rmskurgyr', 'rmskurmag',
             'rmskurliacc','RMSKur')
    plotdata(ID, rmsskeacc, rmsskegyr, rmsskemag, rmsskeliacc, Type, 'rmsskeacc', 'rmsskegyr', 'rmsskemag',
             'rmsskeliacc','RMSSke')

    plotdata1(ID,MaxAx,MaxAy, MaxAz, Type,'MaxAx','MaxAy','MaxAz','MaxAcc')
    plotdata1(ID, MaxGx, MaxGy, MaxGz, Type, 'MaxGx', 'MaxGy', 'MaxGz','MaxGyr')
    plotdata1(ID, MaxMx, MaxMy, MaxMz, Type, 'MaxMx', 'MaxMy', 'MaxMz','MaxMag')
    plotdata1(ID, MaxLiAx, MaxLiAy, MaxLiAz, Type, 'MaxLiAx', 'MaxLiAy', 'MaxLiAz','MaxLiAcc')

    plotdata1(ID, MinAx, MinAy, MinAz, Type, 'MinAx', 'MinAy', 'MinAz','MinAcc')
    plotdata1(ID, MinGx, MinGy, MinGz, Type, 'MinGx', 'MinGy', 'MinGz','MinGyr')
    plotdata1(ID, MinMx, MinMy, MinMz, Type, 'MinMx', 'MinMy', 'MinMz','MinMag')
    plotdata1(ID, MinLiAx, MinLiAy, MinLiAz, Type, 'MinLiAx', 'MinLiAy', 'MinLiAz','MinLiAcc')

    plotdata1(ID, MeAx, MeAy, MeAz, Type, 'MeAx', 'MeAy', 'MeAz','MeAcc')
    plotdata1(ID, MeGx, MeGy, MeGz, Type, 'MeGx', 'MeGy', 'MeGz','MeGyr')
    plotdata1(ID, MeMx, MeMy, MeMz, Type, 'MeMx', 'MeMy', 'MeMz','MeMag')
    plotdata1(ID, MeLiAx, MeLiAy, MeLiAz, Type, 'MeLiAx', 'MeLiAy', 'MeLiAz','MeLiAcc')

    plotdata1(ID, MedAx, MedAy, MedAz, Type, 'MedAx', 'MedAy', 'MedAz', 'MedAcc')
    plotdata1(ID, MedGx, MedGy, MedGz, Type, 'MedGx', 'MedGy', 'MedGz','MedGyr')
    plotdata1(ID, MedMx, MedMy, MedMz, Type, 'MedMx', 'MedMy', 'MedMz','MedMag')
    plotdata1(ID, MedLiAx, MedLiAy, MedLiAz, Type, 'MedLiAx', 'MedLiAy', 'MedLiAz','MedLiAcc')

    plotdata1(ID, VarAx, VarAy, VarAz, Type, 'VarAx', 'VarAy', 'VarAz','VarAcc')
    plotdata1(ID, VarGx, VarGy, VarGz, Type, 'VarGx', 'VarGy', 'VarGz','VarGyr')
    plotdata1(ID, VarMx, VarMy, VarMz, Type, 'VarMx', 'VarMy', 'VarMz','VarMag')
    plotdata1(ID, VarLiAx, VarLiAy, VarLiAz, Type, 'VarLiAx', 'VarLiAy', 'VarLiAz','VarLiAcc')

    plotdata1(ID, StandardDeviationAx, StandardDeviationAy, StandardDeviationAz, Type, 'StandardDeviationAx', 'StandardDeviationAy', 'StandardDeviationAz', 'StandardDeviationAcc')
    plotdata1(ID, StandardDeviationGx, StandardDeviationGy, StandardDeviationGz, Type, 'StandardDeviationGx', 'StandardDeviationGy', 'StandardDeviationGz','StandardDeviationGyr')
    plotdata1(ID, StandardDeviationMx, StandardDeviationMy, StandardDeviationMz, Type, 'StandardDeviationMx', 'StandardDeviationMy', 'StandardDeviationMz','StandardDeviationMag')
    plotdata1(ID, StandardDeviationLiAx, StandardDeviationLiAy, StandardDeviationLiAz, Type, 'StandardDeviationLiAx', 'StandardDeviationLiAy', 'StandardDeviationLiAz','StandardDeviationLiAcc')

    plotdata1(ID, KurtosisAx, KurtosisAy, KurtosisAz, Type, 'KurtosisAx', 'KurtosisAy', 'KurtosisAz','KurtosisAcc')
    plotdata1(ID, KurtosisGx, KurtosisGy, KurtosisGz, Type, 'KurtosisGx', 'KurtosisGy', 'KurtosisGz','KurtosisGyr')
    plotdata1(ID, KurtosisMx, KurtosisMy, KurtosisMz, Type, 'KurtosisMx', 'KurtosisMy', 'KurtosisMz','KurtosisMag')
    plotdata1(ID, KurtosisLiAx, KurtosisLiAy, KurtosisLiAz, Type, 'KurtosisLiAx', 'KurtosisLiAy', 'KurtosisLiAz','KurtosisLiAcc')

    plotdata1(ID, SkeAx, SkeAy, SkeAz, Type, 'SkeAx', 'SkeAy', 'SkeAz','SkeAcc')
    plotdata1(ID, SkeGx, SkeGy, SkeGz, Type, 'SkeGx', 'SkeGy', 'SkeGz','SkeGyr')
    plotdata1(ID, SkeMx, SkeMy, SkeMz, Type, 'SkeMx', 'SkeMy', 'SkeMz','SkeMag')
    plotdata1(ID, SkeLiAx, SkeLiAy, SkeLiAz, Type, 'SkeLiAx', 'SkeLiAy', 'SkeLiAz','SkeLiAcc')