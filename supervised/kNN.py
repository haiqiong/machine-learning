from numpy import *
import operator

#test set.
def createDataSet():
    group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return(group, labels)

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    
    #return the indices of the sorted array.
    sortedDistIndicies = distance.argsort()

    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    #decompose classCount into a list of tuples and then sort the tuples
    #by the second item by itemgetter from operator module.
    sortedClassCount = sorted(classCount.iteritems(), key =
            operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]



