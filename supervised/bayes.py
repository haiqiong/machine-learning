'''
filter inappropriate message using negative or abusive language.
given a vocabulary, transform each document into a vector from the vocabulary.
two categories: abuse(1) and not abuse(0). 
'''

from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', \
                    'please'], 
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', \
                    'stupid'],
                   ['my', 'dalmation', 'is', 'no', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', \
                    'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

#create a list of all the unique words in all training set.
#training set is a list of lists, each list is a post.
def createVocList(dataSet):
    #create an empty set
    vocSet = set([])
    for doc in dataSet:
        #| is set union operator
        vocSet = vocSet | set(doc)
    return list(vocSet)
 
'''
return a vector of 1s and 0s to represent whether a word from the vocabulary
is present or not in the given document.
once occurrence for each word.
'''
def setWordsVec(vocList, inputList):
    #create a vector of all 0s
    resultVec = [0] * len(vocList)
    for word in inputList:
        if word in vocList:
            resultVec[vocList.index(word)] = 1
        else:
            print 'the word: %s is not in the vocabulary.' % word
    return resultVec

'''
trainMat is a list of wordVec.
'''
def trainNB0(trainMat, trainClass):
    numTrainDoc = len(trainMat)
    numWords = len(trainMat[0])
    probAbuse = sum(trainClass)/float(numTrainDoc)
    #for p(wi|ci). changes to ones() from zeros.
    #to avoid 0 in p(w0|1)p(w2|1).., initialize all of occurrence to 1, and
    #initialize the denominators to 2.
    p0Numerator = ones(numWords)
    p1Numerator = ones(numWords)
    #change to 2.0 from 0.0
    p0Denominator = 2.0
    p1Denominator = 2.0
    
    for i in range(numTrainDoc):
        if trainClass[i] == 1:
            #matrix +
            p1Numerator += trainMat[i]
            p1Denominator += sum(trainMat[i])
        else:
            p0Numerator += trainMat[i]
            p0Denominator += sum(trainMat[i])
    #element-wise division
    #p1vec is the prob of the words from the vocabulary under the class 1.
    #change to log()
    p1Vec = log(p1Numerator/p1Denominator)
    p0Vec = log(p0Numerator/p0Denominator)
    return p0Vec, p1Vec, probAbuse

'''
vec2Classify: a vector to classify. It is word vec.
other 3 para are the prob obrained from trainNB0()
'''
def classifyNB(vec2classify, p0vec, p1vec, pClass1):
    p1 = sum(vec2classify * p1vec) + log(pClass1)
    p0 = sum(vec2classify * p0vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    
def testingNB():
    posts, labels = loadDataSet()
    vocList = createVocList(posts)
    trainMat = []
    for postDoc in posts:
        trainMat.append(setWordsVec(vocList, postDoc))
    p0v, p1v, pAb = trainNB0(array(trainMat), array(labels))
    testEntry = ['love', 'my', 'dalmation']
    doc = array(setWordsVec(vocList, testEntry))
    print testEntry, 'classified as:', classifyNB(doc, p0v, p1v, pAb)
    testEntry = ['stupid', 'garbage']
    doc = array(setWordsVec(vocList, testEntry))
    print testEntry, 'classified as:', classifyNB(doc, p0v, p1v, pAb)  
    
'''
multiple occurrences for each word.
'''
def setMultiWordsVec(vocList, inputList):
    resultVec = [0] * len(vocList)
    for word in inputList:
        if word in vocList:
            resultVec[vocList.index(word)] += 1
    return resultVec

'''
remove punctuations and keep strings with len larger than 2
'''
def textParse(bigString):
    import re
    tokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in tokens if len(tok) > 2]

'''
read 25 spam emails and 25 ham emails. choose 10 emails randomly as test set, 
the remaining as training set. evaluate the accuracy of the naive Bayes 
classifier.
'''
def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        wordList = textParse(open('../dataset/email/spam/%d.txt' %i).read())
        docList.append(wordList)
        #fullText is added by extend.
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('../dataset/email/ham/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
        
    vocList = createVocList(docList)
    trainingSet = range(50); testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(randIndex)
        del(trainingSet[randIndex])
    print 'testSet: ', testSet
    
    trainMat = []; trainClasses = []
    for i in trainingSet:
        trainMat.append(setWordsVec(vocList, docList[i]))
        trainClasses.append(classList[i])
        
    p0v, p1v, pAb = trainNB0(array(trainMat), array(trainClasses))
    
    errorCount = 0.0
    for i in testSet:
        testWordVec = array(setWordsVec(vocList, docList[i]))
        testClass = classifyNB(testWordVec, p0v, p1v, pAb)
        if testClass != classList[i]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount) / len(testSet)
    
'''
get the common words in two RSS feeds.
'''
def calMostFreq(vocList, fullText):
    import operator
    freqDict = {}
    for token in vocList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), \
                        reverse=True)
    return sortedFreq[:30]


def localWords(feed0, feed1):
    import feedparser
    docList = []; fullText = []; classList = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        #access one feed
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)  
        
    vocList = createVocList(docList)
    top30Words = calMostFreq(vocList, fullText)
    
    #remove 30 most freq words from vocList
    for freqWord in top30Words:
        if freqWord[0] in vocList:
            vocList.remove(freqWord[0])
    
            
    trainingSet = range(2 * minLen); testSet = []
    #print 'traningSet=', trainingSet, 'minLen', minLen
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        #print 'randIndex=', randIndex
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
        
    trainMat = []; trainClass = []
    for docIndex in trainingSet:
        trainMat.append(setMultiWordsVec(vocList, docList[docIndex]))
        trainClass.append(classList[docIndex])
        
    p0v, p1v, pAb = trainNB0(array(trainMat), array(trainClass))
    errorCount = 0.0
    for docIndex in testSet:
        testWordVec = array(setMultiWordsVec(vocList, docList[docIndex]))
        testClass = classifyNB(testWordVec, p0v, p1v, pAb)
        if testClass != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount)/len(testSet)
    return vocList, p0v, p1v
                                                                 

def testWords():
    import feedparser
    ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    #vocList, pSF, pNY=localWords(sf, ny) 
    getTopWords(sf, ny)

def getTopWords(sf, ny):
    import operator
    vocList, pSF, pNY = localWords(sf, ny)
    topNY = []; topSF = []
    for i in range(len(pSF)):
        if pSF[i] > -6.0:
            topSF.append((vocList[i], pSF[i]))
        if pNY[i] > -6.0:
            topNY.append((vocList[i], pNY[i]))  
    sortedSF = sorted(topSF, key=lambda pair:pair[1], reverse=True)
    #word[0] is unicode, convert it to ascii.
    hotSFWords = [word[0].encode('ascii','ignor') for word in sortedSF]
    print 'SF:', hotSFWords
    sortedNY = sorted(topNY, key=lambda pair:pair[1], reverse=True)
    hotNYWords = [word[0].encode('ascii','ignore') for word in sortedNY]
    print 'NY:', hotNYWords
        
    
    
        

        
        