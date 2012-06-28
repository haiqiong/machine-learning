'''
decision tree
'''

from math import log
import operator

#entropy is about the information of target value.
def calEntropy(dataSet):
    numEntries = len(dataSet)
    labelCount = {}
    for featVec in dataSet:
        #get the target value
        curLabel = featVec[-1]
        if curLabel not in labelCount.keys():
            labelCount[curLabel] = 0
        labelCount[curLabel] +=1
    entropy = 0.0
    #iterate each key of labelCount
    #print 'dict: ', labelCount
    #both iterations work
    #for k, v in labelCount.iteritems():
        #prob = float(v)/numEntries
    for key in labelCount:
        prob = float(labelCount[key])/numEntries    
        # log2(prob)
        entropy -= prob * log(prob, 2)
    return entropy

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    # two features.
    labels = ['no suffacing', 'flippers']
    return dataSet, labels

'''
dataset splitting on a given feature.
axis: the feature col, value: the col divided by two groups, 
equal to value, or not equal to value.
The feature col is removed after being used to split.
return the dataset equals to value.
'''
def splitDataSet(dataSet, axis, value):
    # create a new list to keep origin dataSet intact.
    resultDataSet = []
    for featureVec in dataSet:
        if featureVec[axis] == value:
            # keep the data in front of the feature col
            resultVec = featureVec[:axis]
            # keep the data behind the feature col
            resultVec.extend(featureVec[axis+1:])
            resultDataSet.append(resultVec)
    return resultDataSet
                             
'''
Split the dataset across every feature to see which split gives the highest
information gains.
calculate entropy, split the dataset, measure the entropy on the split sets,
and see if splitting it was the right thing to do.
apply this for all features to determine the best feature to split on.
split on the best feature, best organize the data.
assumption:
1. dataset is a list of lists, all the lists are of equal size.
2. the last col of dataset is the class labels.
'''
def chooseFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # get the feature col
        featureCol = [example[i] for example in dataSet]
        uniqueVals = set(featureCol)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = float(len(subDataSet))/len(featureCol)
            newEntropy = prob * calEntropy(subDataSet)
        
        # the higher the entropy, the more mixed up the data.
        # large infoGain -> lower newEntropy
        # infoGain is the reduction in entropy or the reduction in messiness
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

'''
split the tree to more than two-way branches. 
terminate: run out of features to split or all instances in a branch are 
the same class. if run out of features but the class labels are not all the 
same, a majority vote is used to decide the class of leaf node.
If all instances have the same class, create a leaf node.
Any data that reaches the leaf node is deemed to belong to the class of that
leaf node.
'''
def majorityCount(classList):
    classCount = {}
    for classVal in classList:
        if classVal not in classCount.keys():
            classCount[classVal] = 0
        classCount[classVal] += 1
    #sort by the value.
    sortedClassCount = sorted(classCount.iteritems(), \
                              key=operator.itermgetter(1), reverse=True)
    #return the first element's value.
    return sortedClassCount[0][0]

'''
build a decision tree by recursively choosing the best splitting.
a decision tree is represented by a dict, tree = {bestFeature: {instances}}
labels: the list of features.
'''
def createDecisionTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    #terminate cond1: all instances in the same class
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #terminate cond2: no more features, apply majorityCount
    if len(dataSet[0]) == 1:
        return majorityCount(classList)
    
    bestFeature = chooseFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    
    decisionTree = {bestFeatureLabel: {}}
    #remaining features
    del(labels[bestFeature])
    
    #similar to splitDataSet()
    featureCol = [example[bestFeature] for example in dataSet]
    uniqueFeatureVals = set(featureCol)
    
    #bestFeature is marked in node, each value is an outgoing tran.
    #tree = {bestFeature: {outgoing trans}}
    #an outgoing tran = {value: subtree}
    for value in uniqueFeatureVals:
        #subLables is a copy of lables. keep the original list intact for 
        #each call of createDecisionTree().
        subLabels = labels[:]
        #iterate all unique values from chosen feature and recursively
        #call createDecisionTree() for each split of dataset.
        decisionTree[bestFeatureLabel][value] = createDecisionTree(\
            splitDataSet(dataSet, bestFeature, value), subLabels)
    
    return decisionTree

#traverse the tree and count the leaf nodes.
#compare two strings use cmp(), not ==
def getNumLeafs(dTree):
    numLeaf = 0
    
    firstKey = dTree.keys()[0]
    secondDict = dTree[firstKey]
    #print ('secondDict: ', secondDict)
    #print ('secondDict type:', type(secondDict).__name__)
    #return for the leaf nodes.
    if cmp(type(secondDict).__name__ ,'dict') != 0:
        return 0
    
    for key in secondDict.keys():
        if cmp(type(secondDict[key]).__name__,'dict') == 0:
            #print 'pass dtree:', secondDict[key]
            numLeaf += getNumLeafs(secondDict[key]) 
        else:
            numLeaf += 1
            
    return numLeaf
    

#traverse the tree and count the time hitting the decision nodes.
'''
the elements in dict are randomly {k: v}.
passed dTree in recursive might be {k1:a, k2: {a, b, c}}. But getTreeDepth()
assume dTree is {k2: {a, b, c}
It is weird that getTreeDepth() works without checking dict before for loop 
and using == to compare two strings.
'''
def getTreeDepth(dTree):
    depth = 0
    firstStr = dTree.keys()[0]
    secondDict = dTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        #depth is the deepest branch.
        if thisDepth > depth:
            depth = thisDepth
    return depth


def testTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers':\
                    {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': \
                    {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


