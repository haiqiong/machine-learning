from numpy import *
from os import listdir
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

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    
    #reopen the file to process each line.
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        #populate returnMat, returnMat is numberofLines X 3
        returnMat[index, :] = listFromLine[0:3]
        # classLabelVector is numberOfLines X 1
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    #get the min and max (1 X 3)from column.
    minValues = dataSet.min(0)
    maxValues = dataSet.max(0)
    ranges = maxValues - minValues
    
    m = dataSet.shape[0]
    #expand minValues (1X3) to m rows.
    normDataSet = dataSet - tile(minValues, (m, 1))
    #/ is element-wise division
    normDataSet = normDataSet / tile(ranges, (m, 1))
    
    return normDataSet, ranges, minValues

#10% data of dataset is used to test the accuracy of the classifier.
def datingClassTest():
    testRatio = 0.10
    datingDataSet, datingLabels = file2matrix('../dataset/datingTestSet.txt')
    normDataSet, ranges, minVals = autoNorm(datingDataSet)
    
    m = normDataSet.shape[0]
    trainStart = int(m * testRatio)
    errorCount = 0.0
    
    for i in range(trainStart):
        # k = 3.
        classifier = classify0(normDataSet[i, :], normDataSet[trainStart:m, :],\
                               datingLabels[trainStart:m], 3)
        print "the classifier is: %s, the real class is: %s" \
              % (classifier, datingLabels[i])
        if (cmp(classifier, datingLabels[i]) != 0):
            errorCount += 1.0
    
    print 'the total error rate is: %f' % (errorCount / float(trainStart))
    
#user interface 
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentGame = float(raw_input('percentage of time spent playing \
    video games?'))
    flyMiles = float(raw_input('frequent fly miles earned per year?'))
    iceCream = float(raw_input('liters of ice cream consumed per year?'))
    
    datingDataSet, datingLabels = file2matrix('../dataset/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataSet)
    queryArr = array([flyMiles, percentGame, iceCream])
    queryNorm = (queryArr - minVals) / ranges
    
    k = 3
    classifier = classify0(queryNorm, normMat, datingLabels, k)
    print 'You will probably like this person: ', classifier
    
def img2vector(filename):
    returnVec = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        #lineStr = fr.readline().strip().split()
        for j in range(32):
            returnVec[0, 32*i+j] = int(lineStr[j])
    return returnVec

#recogonize the number 0-9 in the binary digital file.
def digitRecognizeTest():
    digitLabels = []
    
    #build up training set.
    trainFileList = listdir('../dataset/digits/trainingDigits')
    m = len(trainFileList)
    # m X 1024
    trainMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainFileList[i]
        fileName = fileNameStr.split('.')[0]
        classLabel = int(fileName.split('_')[0])
        digitLabels.append(classLabel)
        trainMat[i, :] = img2vector('../dataset/digits/TrainingDigits/%s' \
                                    % fileNameStr)
    
    #build up test set
    testFileList = listdir('../dataset/digits/testDigits')
    mTest = len(testFileList)
    errorCount = 0.0
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileName = fileNameStr.split('.')[0]
        testClassLabel = int(fileName.split('_')[0])
        testVec = img2vector('../dataset/digits/testDigits/%s' % fileNameStr)
        
        k = 3
        classifier = classify0(testVec, trainMat, digitLabels, k)
        print 'the classifier is: %s, the real class is: %s' \
              % (classifier, testClassLabel)
        if (cmp(classifier, testClassLabel) != 0):
            errorCount += 1.0
    print '\nthe total number of errors is: %d' % errorCount
    print '\nthe total error rate is: %f' % (errorCount/float(mTest))