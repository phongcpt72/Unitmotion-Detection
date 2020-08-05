import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.collections import PolyCollection
#
# fs = 11240.
# t = 10
# time = np.arange(fs*t) / fs
# frequency = 1000.
# mysignal = np.sin(2.0 * np.pi * frequency * time)
#
# nperseg = 2**14
# noverlap = 2**13
# f, t, Sxx = signal.spectrogram(mysignal, fs, nperseg=nperseg,noverlap=noverlap)
#
# myfilter = (f>800) & (f<1200)
#
# f = f[myfilter]
# Sxx = Sxx[myfilter, ...]
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# ax.plot_surface(f[:, None], t[None, :], 10.0*np.log10(Sxx), cmap=cm.coolwarm)
# plt.show()


import dtw
def colloc(titlename,numberfilter):
    va = pd.read_csv('Plot/' + titlename + '.csv')
    Type = va.Type

    IsSave = va.loc[(va['Type'] == numberfilter)]

    return IsSave
if __name__ == '__main__':
    nf = 11
    savename = 'PutPhone'

    namefile = 'Calling'
    Calling = colloc(namefile,nf)

    namefile = 'Pocket'
    Pocket = colloc(namefile, nf)

    namefile = 'Swinging'
    Swinging = colloc(namefile, nf)

    namefile = 'Texting'
    Texting = colloc(namefile, nf)

    aa = pd.concat([Calling,Pocket,Swinging,Texting], axis = 0)
    aa.to_csv('Plot/test1.csv', index=False)

    va = pd.read_csv('Plot/test1.csv')

    Pose = va.Pose
    posename = []
    for i in range(len(Pose)):
        if Pose[i] == 1:
            posename.append('Calling')
        elif Pose[i] == 2:
            posename.append('Pocket')
        elif Pose[i] == 3:
            posename.append('Swinging')
        elif Pose[i] == 4:
            posename.append('Texting')

    va['PoseName'] = posename
    idarr = []
    sums = 0
    for i in range(len(Pose)):
        idarr.append(sums)
        sums+=1

    va['Timestamp'] = idarr

    va.to_csv('Plot/' + savename + '.csv', index=False)


    print('Done')

