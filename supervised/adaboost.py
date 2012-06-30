from numpy import *

def loadSimpleData():
    dataMat = matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def stumpClassify(dataMat, dimen, threshVal, ineq):
    retArr = ones((dataMat.shape[0], 1))
    if ineq == 'lt':
        retArr[dataMat[:, dimen] <= threshVal] = -1.0
    else:
        retArr[dataMat[:, dimen] > threshVal] = -1.0
    return retArr

'''
bestStump: being used to store the classifier information corresponding to 
the best choice of a decision stump given the weight vector D.
numSteps: being used to iterate over the possible values of the fatures.
The second for loop: it might make sense to set the threshold outside
the extremes of your range, so there ae two extra steps outside the range.
'''
def buildStump(dataArr, classLabels, D):
    dataMat = mat(dataArr); labelMat = mat(classLabels).T
    m, n = dataMat.shape
    numSteps = 10.0; bestStump = {}; bestClassEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = min(dataMat[:, i]); rangeMax = max(dataMat[:, i])
        stepSize = (rangeMax - rangeMin) / numSteps

        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + float(j) * stepSize
                predicatedVal = stumpClassify(dataMat, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                #for the row that predicatedVal == labelMat, errArr[row] = 0
                errArr[predicatedVal == labelMat] = 0
                weightedError = D.T * errArr
                #print 'split: dim %d, thesh %.2f, ineqal: %s, \
                #weighted error:%.3f' %(i,threshVal,inequal,weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predicatedVal.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
                    
    return bestStump, minError, bestClassEst   

'''
DS: decision stump
1. calculate alpha
alpha tells the total classifier how much to weight the output from the stump.
max(error, 1e-16) to make sure you don't have a divide-by-zero error in the 
case where there's no error.
2. calculate D
multiply(): element-wise product.
exp(array): return an array where each element is calculated by exp.
3. aggClassEst.
It is a floating point number, to get the binary class, use the sign() function. 
'''
def adaBoostDS(dataArr, classLabels, numIter = 40):
    bestStumpArr = []
    m = dataArr.shape[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIter):
        bestStump, error, bestClassEst = buildStump(dataArr, classLabels, D)
        print 'D:', D.T
        alpha = float(0.5 * log((1.0 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        bestStumpArr.append(bestStump)
        print 'ClassEst:', bestClassEst.T
        
        #multiply(): element-wise product. class real result X estimation
        expon = multiply(-1 * alpha * mat(classLabels).T, bestClassEst)
        #exp(expon): calculate exp for each element in mat expon
        D = multiply(D, exp(expon)) / sum(D)
        
        #aggClassEst is float mat.
        aggClassEst += alpha * bestClassEst
        print 'aggClassEst:', aggClassEst
        
        #aggClassEst is float mat, use its sign to compare with mat classLabels
        aggError = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))
        errorRate = sum(aggError)/m
        print 'total error:', errorRate
        
        if errorRate == 0.0:
            break
        
    return bestStumpArr

def adaClassify(dataToClass, classifierArr):
    dataMat = mat(dataToClass)
    m = dataMat.shape[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'], \
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return sign(aggClassEst)
                                                
# thes last col is class label.                                              
def loadDataSet(fileName):
    numFeature = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineList = []
        curLine = line.strip().split('\t')
        for i in range(numFeature - 1):
            lineList.append(float(curLine[i]))
        dataMat.append(lineList)
        labelMat.append(float(curLine[-1]))
    return array(dataMat), labelMat

