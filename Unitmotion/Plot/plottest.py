# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import matplotlib.cm as cmx
# from pandas import read_csv
# import dtw
#
#
# #df = read_csv('iris.csv')
# df = read_csv('DownStair.csv')
#
# # map Name to integer
# #pos = df.loc[:,["Name"]].groupby("Name").count().reset_index()
# pos = df.loc[:,["PoseName"]].groupby("PoseName").count().reset_index()
#
# # create a new column in the dataframe which contains the numeric value
# tag_to_index = lambda x: pos.loc[pos.PoseName == x.PoseName].index[0]
# df.loc[:,"name_index"]=df.loc[:,["PoseName"]].apply(tag_to_index, axis=1)
#
# # Set the color map to match the number of species
# hot = plt.get_cmap('hot')
# cNorm  = colors.Normalize(vmin=0, vmax=len(pos))
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)
#
# # Get unique names of species
# for (name, group) in df.groupby("name_index"):
#     plt.scatter(group.Timestamp, group.VarianceGz, s=15, label=pos.iloc[name].get("PoseName"), color=scalarMap.to_rgba(name))
#
# plt.xlabel('Timestamp')
# plt.ylabel('MaxGy')
# plt.title('Passing The Door - Holding Style')
# plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from dtwalign import dtw

np.random.seed(1234)
# test data
x = np.sin(2*np.pi*3.1*np.linspace(0,1,101))
x += np.random.rand(x.size)
y = np.sin(2*np.pi*3*np.linspace(0,1,120))
y += np.random.rand(y.size)
res = dtw(x,y)

print(len(x))
print(len(y))

print("dtw distance: {}".format(res.distance))
print("dtw normalized distance: {}".format(res.normalized_distance))

plt.plot(x,label="query")
plt.plot(y,label="reference")
plt.legend()
#plt.ylim(-1,3)
plt.show()



# dtw distance


"""
if you want to calculate only dtw distance (i.e. no need to gain alignment path),
give 'distance_only' argument as True (it makes faster).
"""
#res = dtw(x,y,distance_only=True)

