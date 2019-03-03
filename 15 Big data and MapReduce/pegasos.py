# The Pegasos SVM algorithm
# Set w to all zeros
# For each batch
#     Choose k data vectors randomly
#     For each vector
#     If the vector is incorrectly classified:
#         Change the weights vector: w
#     Accumulate the changes to w
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    dataMat, labelMat = [], []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return np.array(dataMat), np.array(labelMat)


def predict(w, x):
    return np.dot(w, x)


def batchPegasos(dataSet, labels, lam, T, k):
    # T and k set the number of iterations and the batch size
    m, n = np.shape(dataSet)
    w = np.zeros(n)
    dataIndex = list(range(m))
    for t in range(1, T + 1):
        wDelta = np.zeros(n)
        eta = 1 / (lam * t)
        np.random.shuffle(dataIndex)
        for j in range(k):
            i = dataIndex[j]
            p = predict(w, dataSet[i])
            if labels[i] * p < 1:
                # Accumulate changes
                wDelta += labels[i] * dataSet[i]
        w = (1 - 1 / t) * w + (eta / k) * wDelta
    return w


datArr, labelList = loadDataSet('testSet.txt')
finalWs = batchPegasos(datArr, labelList, 2, 50, 100)
print(finalWs)

plt.figure()
x1, y1 = [], []
xm1, ym1 = [], []
for i in range(len(labelList)):
    if labelList[i] == 1.0:
        x1.append(datArr[i, 0])
        y1.append(datArr[i, 1])
    else:
        xm1.append(datArr[i, 0])
        ym1.append(datArr[i, 1])
plt.scatter(x1, y1, marker='s', s=80, edgecolor="black")
plt.scatter(xm1, ym1, marker='o', s=50, c='red', edgecolor="black")
x = np.arange(-6.0, 8.0, 0.1)
y = -finalWs[0] * x / finalWs[1]
y2 = -0.55309587 * x / -0.04097803
plt.plot(x, y)
plt.plot(x, y2, 'g-.')
plt.axis([-6, 8, -4, 5])
plt.legend(('50 Iterations', '2 Iterations'))
plt.show()
