from numpy import *

def loadDataSet(filename):
    dataSet = []; labels = []
    fr = open(filename)
    for line in fr.readlines():
        lineList = line.strip().split('\t')
        dataSet.append([float(lineList[0]), float(lineList[1])])
        labels.append(float(lineList[2]))
    return dataSet, labels

#randomly select one integer from a range that is not equal to i.
def selectRand(i, m):
    j = i
    while (j==i):
        j = int(random.uniform(0, m))
    return j

#clip alpha that greater than H or less than L.
def clipAlpha(alpha, H, L):
    if alpha > H:
        return H
    if alpha < L:
        return L
    
def smoSimple(dataMatIn, labelsIn, toler, maxIter):
    