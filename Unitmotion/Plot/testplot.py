from numpy import where
from collections import Counter
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import numpy as np
# define dataset
X, y = make_blobs(n_samples=1000, centers=4, random_state=1)

va = read_csv('Walking.csv')
MaxAx = va.MaxAx
MinAy = va.MinAy
MaxGx = va.MaxGx
tmparr = np.zeros_like(MaxAx)


aa = pd.DataFrame({"MaxAx": MaxAx, "MinAy": MinAy}) # x,y in 2D
y = va.PoseName
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
plt.figure(1)
plt.xlabel('Max Ax')
plt.ylabel('Min Ay')
plt.title('Walking')
plt.legend()

plt.show()



va = read_csv('Upstair.csv')
MaxAx = va.MaxAx
MinAy = va.MinAy
MaxGx = va.MaxGx
tmparr = np.zeros_like(MaxAx)


aa = pd.DataFrame({"MaxAx": MaxAx, "MinAy": MinAy}) # x,y in 2D
y = va.PoseName
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

plt.xlabel('Max Ax')
plt.ylabel('Min Ay')
plt.title('UpStair')
plt.legend()





va = read_csv('DownStair.csv')
MaxAx = va.MaxAx
MinAy = va.MinAy
MaxGx = va.MaxGx
tmparr = np.zeros_like(MaxAx)


aa = pd.DataFrame({"MaxAx": MaxAx, "MinAy": MinAy}) # x,y in 2D
y = va.PoseName
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

plt.xlabel('Max Ax')
plt.ylabel('Min Ay')
plt.title('DownStair')
plt.legend()





va = read_csv('PassingTheDoor.csv')
MaxAx = va.MaxAx
MinAy = va.MinAy
MaxGx = va.MaxGx
tmparr = np.zeros_like(MaxAx)


aa = pd.DataFrame({"MaxAx": MaxAx, "MinAy": MinAy}) # x,y in 2D
y = va.PoseName
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

plt.xlabel('Max Ax')
plt.ylabel('Min Ay')
plt.title('Detect Holding Styles')
plt.legend()
plt.show()
