from numpy import *

'''
can use array or mat for dataset.
if use array, be careful of slicing.
arr[:, 1] makes 2d array become 1d array. if still need 2d array, we should
arr[:, [1]]
'''
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineList = line.strip().split('\t')
        #map(func, sequence)
        lineFloat = map(float, lineList)
        dataMat.append(lineFloat)
    #return mat(dataMat)
    return array(dataMat)
        
#distance metric: Eclud measure for two vectors.
def distEclud(vecA, vecB):
    #** for array. vecA and vecB are mat(m, 1). A is the array rep. of a mat.
    return (sum((vecA - vecB) ** 2)) ** 0.5
    # for mat, we should 
    #return (sum((vecA - vecB).A ** 2)) ** 0.5
    #return sqrt(sum(power(vecA - vecB, 2)))

'''
centroid is the center of a cluster.
return an array.
'''
def randCent(dataSet, k):
    n = dataSet.shape[1]
    centroids = zeros((k,n))
    #centroids = mat(zeros((k, n)))
    
    for j in range(n):
        # minJ is a float number
        minJ = float(min(dataSet[:, j]))
        rangeJ = float(max(dataSet[:, j] - minJ))
        
        #numpy.random.rand(d0, d1, ..., dn), d is dimension. return (0, 1).
        #polpulate col j
        centroids[:, [j]] = minJ + rangeJ * random.random((k, 1))
        
    print 'rand centoid:', centroids
    return centroids

'''
1. start from rand centroid for each cluster.
2. for each point, calculate dist to each centroid, assign the cluster by the
shortest dist.
3. update centroid as the mean of value of the points(a row) in a cluster.
4. repeat 2-3 until points stop changing cluster.
'''
def kmeans(dataSet, k, distMetric=distEclud, creatCent=randCent):
    m = dataSet.shape[0]
    #(cluster id, dist) for each point. Array is enough, not necessary for mat
    clusterAssign = zeros((m, 2))
    clusterChange = True
    centroids = creatCent(dataSet, k)
    iteration = 0
    
    while clusterChange:
        iteration += 1
        clusterChange = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJ = distMetric(centroids[j, :], dataSet[i, :])
                if distJ < minDist:
                    minDist = distJ; minIndex = j
            if clusterAssign[i, 0] != minIndex:
                clusterChange = True
            clusterAssign[i, :] = minIndex, minDist ** 2
        #print centroids
        #update centroids
        for centerId in range(k):
            #[0] get the index col of clusterAssign
            indices = nonzero(clusterAssign[:, 0] == centerId)[0]
            #print indices
            pointsInCluster = dataSet[indices]
            #axis = 0 by col, mean for each of col.
            centroids[centerId, :] = pointsInCluster.mean(axis=0)
    return centroids, clusterAssign

def kmeanTest():
    datMat = loadDataSet('../dataset/kmean-testSet.txt')
    print min(datMat[:, 0])
    print max(datMat[:, 0])
    print min(datMat[:, 1])
    print max(datMat[:, 1])
    
    #check if randCent() produces a value between min and max
    randCent(datMat, 2)
    
    #distance between col0 and col1
    print distEclud(datMat[0], datMat[1])
    
    print 'centroids:'
    centroids, clusterAssign= kmeans(datMat, 4)
    print centroids
    
'''
bisecting k-means starts from one cluster and then split into two. The cluster 
to split is decided by minimizig the SSE(sum of square error). The splitting is
repeated until the user-defined number of clusters.
has bugs.
 ''' 
def bikmeans(dataset, k, distMetric=distEclud):
    m = dataset.shape[0]
    clusterAssign = zeros((m, 2))       
    #init center is the mean of col0
    #tolist() convert an array to python list
    initCent = mean(dataset, axis=0).tolist()[0] 
    centroid0 = tile(array(initCent), (1, dataset.shape[1]))
    centList = [centroid0]
    for j in range(m):
        clusterAssign[j,1] = distMetric(dataset[j, :], centroid0) ** 2
    
    while (len(centList) < k):
        lowestSSE = inf
        #split each cluster to find out the one with min error
        for i in range(len(centList)):
            pointsInCluster = dataset[nonzero(clusterAssign[:, 0]==i)[0]]
            splitCents, splitClusterAssign = kmeans( \
                pointsInCluster, 2, distMetric)
            splitSSE = sum(splitClusterAssign[:, 1])
            noSplitSSE = sum(clusterAssign[nonzero(clusterAssign[:, 0] != i)[0]])
            print 'splitSSE: %f, noSplitSSE: %f' % (splitSSE, noSplitSSE)
            if splitSSE + noSplitSSE < lowestSSE:
                bestSplitCents = splitCents
                bestSplitCluster = i
                bestSplitClusterAssign = splitClusterAssign.copy()
                lowestSSE = splitSSE + noSplitSSE
        
        #update splitted cluster assign
        #the clust is split by 2 with label 0, 1. change the label to cluster 
        #number and the new cluster to be added.
        bestSplitClusterAssign[nonzero(bestSplitClusterAssign[:, 0] == 1)[0]] \
            = bestSplitCluster
        bestSplitClusterAssign[nonzero(bestSplitClusterAssign[:, 0] == 0)[0]] \
                    = len(centList)
        #update centList
        centList[bestSplitCluster] = bestSplitCents[0, :]
        centList.append(bestSplitCents[1, :])
        clusterAssign[nonzero(clusterAssign[:, 0] == bestSplitCluster)[0]] = \
            bestSplitClusterAssign
    return array(centList), clusterAssign
                                  
#convert address to long and lat by Yahoo! PlaceFinder API
import urllib
import json
def geoGrab(stAddr, city):
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'
    params['appid'] = 'Mw8WM432'
    params['location'] = '%s %s' % (stAddr, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params
    print yahooApi
    c = urllib.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def addPlaceFind(fileName):
    fw = open('../dataset/clubplace.txt', 'w')
    for line in open(fileName).readlines():
        #col0:name, col1: addr, col2: city, state
        line = line.strip()
        lineAddr = line.split('\t')
        retDict = geoGrab(lineAddr[1], lineAddr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            longt = float(retDict['ResultSet']['Results'][0]['longitude'])
            print '%s \t %f \t %f' % (lineAddr[0], lat, longt)
            fw.write('%s \t %f \t %f' % (line, lat, longt))
        else:
            print 'error fetching'
        #delay addPlaceFind() for 1 sec. This ensures not to make too many
        #API calls too quickly. Otherwise, will be blocked.
        sleep(1)
    fw.close()
 
'''
lat and longt are given by degree, we need radians for sin() and cos().
use spherical law of cosines to calculate the distance for two ponints on the 
earth.
vecA=[lat, longt]
'''
def distSLC(vecA, vecB):  
    a = sin(vecA[0, 0] * pi / 180) * sin(vecB[0, 0] * pi / 180)
    b = cos(vecA[0, 0] * pi / 180) * cos(vecB[0, 0] * pi / 180) * \
        cos((vecB[0, 1] - vecA[0, 1]) * pi / 180)
    return arccos(a + b) * 6371.0
     