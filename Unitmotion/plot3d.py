from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

eva = pd.read_csv('machine/overlap80_2s/newgroup317/Non-Moving.csv')

stand = eva[(eva['Unitmotion'] == 'Standing')]
stand = stand.reset_index(drop=True)

pot = eva[(eva['Unitmotion'] == 'Phone on table')]
pot = pot.reset_index(drop=True)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = stand.MaxAx
ys = stand.MaxAy
zs = stand.MaxAz

xt = pot.MaxAx
yt = pot.MaxAy
zt = pot.MaxAz

ax.scatter(xs, ys, zs, c='r', marker='o', label='Standing')
ax.scatter(xt, yt, zt, c='b', marker='^', label='Phone on table')

ax.set_xlabel('Max Ax')
ax.set_ylabel('Max Ay')
ax.set_zlabel('Max Az')
plt.title('Non-Moving', fontweight='bold')
plt.legend()
plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

eva = pd.read_csv('machine/overlap80_2s/newgroup317/HoldingStyles.csv')

po = eva[(eva['Pose'] == 'Pocket')]
po = po.reset_index(drop=True)

sw = eva[(eva['Pose'] == 'Swinging')]
sw = sw.reset_index(drop=True)

te = eva[(eva['Pose'] == 'Texting')]
te = te.reset_index(drop=True)

ca = eva[(eva['Pose'] == 'Calling')]
ca = ca.reset_index(drop=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xp = po.MaxAx
yp = po.MaxAy
zp = po.MaxAz

xs = sw.MaxAx
ys = sw.MaxAy
zs = sw.MaxAz

xc = ca.MaxAx
yc = ca.MaxAy
zc = ca.MaxAz

xt = te.MaxAx
yt = te.MaxAy
zt = te.MaxAz

ax.scatter(xp, yp, zp, color='#AD4F0E', marker='o', label='Pocket')
ax.scatter(xt, yt, zt, color='#2d7f5e', marker='^', label='Texting')
ax.scatter(xc, yc, zc, color='#11E8D4', marker='v', label='Calling')
ax.scatter(xs, ys, zs, color='#557f2d', marker='*', label='Swinging')

ax.set_xlabel('Max Ax')
ax.set_ylabel('Max Ay')
ax.set_zlabel('Max Az')
plt.title('Holding Styles', fontweight='bold')
plt.legend()
plt.show()

