#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from sklearn import svm
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]


#########################################################
### your code goes here ###
t0 = time()
clf = svm.SVC(C=10000, kernel="rbf")
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")
#accuracy = clf.score(features_test, labels_test)
#print(accuracy)
#ans10 = clf.predict([features_test[10]])
#ans26 = clf.predict([features_test[26]])
#ans50 = clf.predict([features_test[50]])
#print(ans10, ans26, ans50)
ans = clf.predict(features_test)

print(list(ans).count(1))
print("training time:", round(time()-t0, 3), "s")
#########################################################


