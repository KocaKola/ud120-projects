# README project ud120

### Introduction
This README.md file includes those information which needed to explain. The process of the machine learning is a sequential combination of several operations, such as data cleaning, feature generation and scaling, parameters tuning and validation.

### Data Exploration
 The total number of the data points is 146 and there're 21 features in each point. Among the features, there're two types, finances and emails, separated as features_list and email_features in the code. Those features which missed less than half of the data size are remained since too much NaN could lead to the overfit problem. Besides, two total value features are dropped, then the left features are poi, salary, bonus, expenses, exercised_stock_options, other, restricted_stock where the poi label included. Also, the email features are remained except the email_address string.

 The numbers of the POI and non-POI are 18 and 128 respectively. The ratio of POI is about 12.33%. 

### Deal with the Outlier
We start with the salary and bonus feature, and it turns out there exists a TOTAL data point which add up all. After cleaning the TOTAL outlier, we go through all the 5 features compared to the salary. It still seems that there're probably 4 more outliers in the dataset. However, these outliers are reasonable data points, it shouldn't be removed.

### Optimize the Features
Firstly, I thought the email features could be used to generate new features. I've rewrite the getToFromStrings function in the poi_flag_email.py to create the from/to poi features, but it turns out that they were provided in the original data dict. :(

So I think, the relations between the features in the original dataset could be used as new features. There are two pairs of features, which are from_messages/from_this_person_to_poi and to_messages/from_poi_to_this_person. The ratio between poi-related emails count and total count could be reasonable features. So the new features are
1. from_poi_to_this_person_ratio = from_poi_to_this_person/to_messages
2. from_this_person_to_poi_ratio = from_this_person_to_poi/from_messages

The two new features are generated and added to the feature list.

### Algorithm tuning
In this section, 3 different algorithms are picked, Gaussian Naive Bayes, Decision Tree and Support Vector Machine. Firstly, the 3 algorithms are employed without any parameters tuning.The Naive Bayes algorithm has 0.82927 accuracy, however, the precision and recall are only 0.28273and 0.184, respectively. The Decision Tree algorithm has a better performance, with 0.82587 accuracy,0.34292 precision and 0.33400 recall. The SVC() doesn't even make a true positiveprediction, so its performance couldn't be reviewed, maybe it should be changed to LinearSVC(), which has a 0.65213 accuracy with low precision and recall. Furthermore, these algorithm should be tuned in the next step. There isn't much we can do with the Naive Bayes algorithm, so it's skipped the tuning step. We mainly focus on the Decision Tree and Support Vector Machine algorithm.

GridSearchCV function is utilized to tune the decision tree and support vector machine.

For the decision tree, the min_samples_split parameter is tuned, which is the minimum number of samples required to split an internal node, affecting the toughness of the fitting curve. The default value is 2, we set a range from 2 to 10, after the grid search, we set the best parameter to the classifier, and rerun the tester function. It turns out that the best parameter should be around 4. The accuracy raises from 0.82687 to 0.83047, the precision is also better than that without tuning. And if we change the parameter to a higher value, the performances will go down because of overfit problem.

For the support vector machine, the C parameter is picked for tuning, with a range from 1 to 10. C is the penalty parameter of the error term, larger C value means a more complex decision boundary. After the grid search, the accuracy raises to 0.64227 from untuned 0.64, also, the precision is better. However, the overall performance of the SVM is poorer than that of the decision tree, so decision tree is chosen as the final classifier. 

All the results are listed in the under table.

|              Methods              | Accuracy | Precision |  Recall |    F1   |    F2   |
|:---------------------------------:|:--------:|:---------:|:-------:|:-------:|:-------:|
| Naive Bayes (GaussianNB, untuned) |  0.82927 |  0.28373  | 0.18400 | 0.22323 | 0.19791 |
|      Decision Tree (untuned)      |  0.82687 |  0.34558  | 0.33400 | 0.33969 | 0.33625 |
|       Decision Tree (tuned)       |  0.83047 |  0.35237  | 0.32400 | 0.33759 | 0.32930 |
|      SVM (LinearSVC, untuned)     |  0.64000 |  0.13315  | 0.30850 | 0.18601 | 0.24418 |
|       SVM (LinearSVC, tuned)      |  0.64227 |  0.13571  | 0.31350 | 0.18943 | 0.24842 |

### Validation and Evaluation

In the section above, several algorithms are tuned for better performance. In the processes, all the features and labels are split into the training set and test set for validation. The cross validation could help to reduce the impact of the overfit problem, thus, we could get a trustworthy accuracy number.

Also, during the tuning process, accuracy, precision and recall are employed as the benchmarks. With these benchmarks, the performance of the algorithms could be precisely qualified. Precision is the ratio of true positives and the sum of true positives and false positives, while recall is the ratio of true positives and the sum of true positives and false negatives.Usually there's a trade-off between precision and recall, we could see that while tuning, the precision goes up with a lower recall. In this case, we want to have a precise poi identifier, so the slightly low recall could be accepted.

### Conclusion

In this project, poi exploration is done with the enron dataset. Firstly, we take a look at the basic information of the dataset. Secondly, we go deeper and visualize the data so the outliers are removed. Then, two new features are generated to present the poi/overall mail ratio. After that, 3 different algorithms are selected to perform the train and test. Untuned and tuned algorithms' performances are compared, with the grid search function, decision tree and svm show better accuracy and precision. And it turns out the decision tree algorithm has the best accuracy and precision. 

### References
1. Project rubics, https://review.udacity.com/#!/rubrics/27/view
2. Built-in Types, Python 2.7.14, https://docs.python.org/2/library/stdtypes.html
3. Making a flat list out of list of lists in Python, https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
4. Support Vector Machine, http://scikit-learn.org/stable/modules/svm.html#svm-classification
5. Naive Bayes, http://scikit-learn.org/stable/modules/naive_bayes.html
6. Decision Tree, http://scikit-learn.org/stable/modules/tree.html
