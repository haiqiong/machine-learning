from numpy import *

#x0: 1, x1, x2 from the txt file.
def loadDataSet():
    dataMat = []; classMat = []
    fr = open('../dataset/testSet.txt')
    for line in fr.readlines():
        lineList = line.strip().split()
        dataMat.append([1.0, float(lineList[0]), float(lineList[1])])
        classMat.append(int(lineList[2]))
    return dataMat, classMat

def sigmoid(inx):
    return 1.0/(1+exp(-inx))

'''
dataMatIn is a 2D Numpy array, where the cols are the features and rows are
the training instances.
alpha is the step size.
maxCycles for ternimation.
'''
def gradAscent(dataMatIn, classMatIn):
    dataMat = mat(dataMatIn)
    classMat = mat(classMatIn).T
    m, n = dataMat.shape
    alpha = 0.001
    maxCycles = 500
    #row is the number of features. n X 1
    weights = ones((n, 1))
    for i in range(maxCycles):
        # m X n * n X 1 = m X 1
        h = sigmoid(dataMat * weights)
        #the difference between actual class and the predicted class
        error = classMat - h
        #n X 1 = n X m * m X 1
        weights = weights + alpha * dataMat.T * error
    return weights
    
#update weights for each instance in the dataset.
def stocGradAscent0(dataMatIn, classMatIn):
    dataMat = mat(dataMatIn)
    classMat = mat(classMatIn).T    
    m,n = dataMat.shape
    alpha = 0.01
    #n X 1
    weights = ones((n, 1))
    for i in range(m):
        #1 X n * n X 1 = 1 X 1
        h = sigmoid(dataMat[i] * weights)
        error = classMat[i] - h
        #n X 1 * 1 X 1
        weights = weights + alpha * dataMat[i].T * error
    return weights

'''
alpha is changed in each iteration.
randomly pick up vector.
'''
def stocGradAscent1(dataMatIn, classMatIn, numIter = 150):
    dataMat = mat(dataMatIn)
    classMat = mat(classMatIn).T    
    m,n = dataMat.shape
    weights = ones((n, 1))
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            #1 X n * n X 1 = 1 X 1
            h = sigmoid(dataMat[randIndex] * weights)
            error = classMat[randIndex] - h
            #n X 1 * 1 X 1
            weights = weights + alpha * dataMat[randIndex].T * error
            del(dataIndex[randIndex])
    return weights

'''
inX and weights are array
the categories are Yes or No.
'''
def classifyVector(inX, weights):
    #element-wise production
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
    
def colicTest():
    frTrain = open('../dataset/horseColicTraining.txt')
    frTest = open('../dataset/horseColicTest.txt')
    trainSet = []; trainLabels = []
    for line in frTrain.readlines():
        lineList = line.strip().split('\t')
        #convert each value of feature to float, 20 features in total.
        values = []
        for i in range(20):
            values.append(float(lineList[i]))
        trainSet.append(values)
        trainLabels.append(float(lineList[-1]))
    testSet = []; testLabels = []

    trainWeights = stocGradAscent1(array(trainSet), trainLabels, 500)
    errorCount = 0; numTestVec = 0.0
    
    for line in frTest.readlines():
        numTestVec += 1.0
        lineList = line.strip().split('\t')
        #convert each value of feature to float, 20 features in total.
        values = []
        for i in range(20):
            values.append(float(lineList[i]))
        if int(classifyVector(array(values), trainWeights)) \
           != int(lineList[-1]):
            errorCount += 1
    errorRate = float(errorCount)/numTestVec
    print 'the error rate is: {}'.format(errorRate)
    return errorRate
                       
    
def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print 'after {} iterations the average error rate \
    is: '.format(numTests, errorSum/float(numTests))