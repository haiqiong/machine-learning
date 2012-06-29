Machine learning using NumPy,
by Haiqiong Yao
6/26/2012

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






