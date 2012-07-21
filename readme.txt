Machine learning using Python and NumPy,
by Haiqiong Yao
6/26/2012

/supervised
1. k-nearest classification
kNN.py
the original version. 
(1) calculate the Euclidian distance from unknown element to each element in the dataset.
(2) get the classes of k elements with the smallest distances.
(3) vote from k classes to generate the class of the unknow element.
(4) read data from txt file and normalize data.
(5) using 10% data to test the accuracy of the classifier.
(6) build a user interface to input the feature of a person and provide the classification of that person.
classifyPerson()
(7) recogonize number 0-9 in the binary images.
convert the binary images to text format.
digitRecognizeTest()  error rate: 0.011628

2. decision tree using algorithm ID3
tree.py
(1) calculate entropy.
(2) find out the best splitting feature to make dataset organized best.
(3) build a decision tree by recursivly choosing the best splitting.
(4) get the number of leaves and the depth.
(5) build a classifier using the decision tree from the training data.
(6) pesisting the decision tree with pickle.

3. document classification with naive Bayes
bayes.py
(1) filter out malicious posts.
Give some posts and their categories(abusive or not abusive), build a naive Bayes classifier. The feature is the presence or absence of a word.
(2) filter out spam emails 
Spam email with naive Bayes. Given 50 emails including normal ones and spam, randomly pick 10 of them as test set and others as training set. Average error rate for 10 times tests: 0.012
(3) display the most common words given in two RSS feeds.
Use feed parser to read RSS feeds.
Address underflow with logarithm of probability. Consider multiple occurrances of one word. 
Average error rate: 0.30
recommend to remove stop words.

4. logistic regression
logRegress.py
make a line to separate the different classes of data. class labels are 0 and 1.
(1) Calculate the weights by an optimization algorithm, gradient ascent.
(2) stochastic gradient ascent updates the weights for each instance in the dataset.
(3) deal with missing values in dataset: replace the missing value with 0 in the training set and throw away the instance in test set.
average error rate for 10 iterations: 0.37293

5. support vector machines
svm.py
class labels are -1 and 1.
(1) use sequential minimal optimization(SMO) to find a set of alpha and b.
(2) imcomplete implementation. SVM is too complicated.

6. AdaBoost
adaboost.py
(1) createe a weak learner with a decision stump.
(2) create AdaBoost to use multiple weak learners.
(3) implement full AdaBoost with the decision stump.
(4) build a adaClassifier.
error rate for horseColicTest2.txt is 0.2388 (16/67)

7. predicate continuous values with regression
regression.py
(1) calculate weights using mean-square error.
(2) locally weighted linear regression.
control k to get best-fit line as straight line or curve line. Be careful of underfitting and overfitting.
expensive computation for using the entire dataset to make one estimate.
(3) to solve dataset not full rank, using shrinkage methods: ridge regression, 

/unsupervised
8. k-mean cluster
kmeans.py 
(1) k is defined by user. group data into k clusters, the center of each cluster is the mean of the values in that cluster. k-mean is effective, but sensitive to the initial cluster placement.
(2) iteratively cluster points. start from one cluster and split the one with lowest error. bikmean creates a better clusters than k-means.
examples:
(1) kmean-testSet.txt. cluster points into k=4 groups by 3 iterations.
(2) kmean-portlandclub.txt. 
given the address of 70 clubs in portland, group the clubs close together.
use yahoo!PlaceFinder API to get lat and longt for a street addr.
calculate earth distance between two points.
use bikmean() to cluster the points.



/recommendation
8. filtering
recommendations.py
(1) finding similar people by calculating similarity score, Eudidean distance or pearson correlation.











