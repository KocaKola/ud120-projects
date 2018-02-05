# README project ud120

### Introduction
This README.md file includes those information which needed to explain. The process of the machine learning is a sequential combination of several operations, such as data cleaning, feature generation and scaling, parameters tuning and validation.

### Data Exploration
 The total number of the data points is 146 and there're 21 features in each point. Among the features, there're two types, finances and emails, separated as features_list and email_features in the code. Those features which missed less than half of the data size are remained since too much NaN could lead to the overfit problem. Besides, two total value features are dropped, then the left features are poi, salary, deferral_payments, total_payments, loan_advances, bonus, restricted_stock_deferred, deferred_income, total_stock_value, expenses, exercised_stock_options, other, long_term_incentive, restricted_stock, director_fees where the poi label included. Also, the email features are remained except the email_address string.

 The numbers of the POI and non-POI are 18 and 128 respectively. The ratio of POI is about 12.33%. The numbers of the NaN existed in each features are described as below.

 |        finance feature    | NaN counts |      email feature      | NaN counts |
|:-------------------------:|:----------:|:-----------------------:|:----------:|
|            poi            |      0     |       to_messages       |     60     |
|           salary          |     51     |      email_address      |     35     |
|     deferral_payments     |     107    | from_poi_to_this_person |     60     |
|       total_payments      |     21     |      from_messages      |     60     |
|       loan_advances       |     142    | from_this_person_to_poi |     60     |
|           bonus           |     64     | shared_receipt_with_poi |     60     |
| restricted_stock_deferred |     128    |                         |            |
|      deferred_income      |     97     |                         |            |
|     total_stock_value     |     20     |                         |            |
|          expense          |     51     |                         |            |
|  exercised_stock_options  |     44     |                         |            |
|           other           |     53     |                         |            |
|    long_term_incentive    |     80     |                         |            |
|      restricted_stock     |     36     |                         |            |


### Deal with the Outlier
We start with the salary and bonus feature, and it turns out there exists a TOTAL data point which add up all. After cleaning the TOTAL outlier, we go through all the 5 features compared to the salary. It still seems that there're probably 4 more outliers in the dataset. However, these outliers are reasonable data points, it shouldn't be removed.

### Optimize the Features
Firstly, I thought the email features could be used to generate new features. I've rewrite the getToFromStrings function in the poi_flag_email.py to create the from/to poi features, but it turns out that they were provided in the original data dict. :(

So I think, the relations between the features in the original dataset could be used as new features. Thus, 3 set of features list are generated for comparison. Firstly, my_feature_list consists of cut-off features list with newly generated features. Secondly, full_features_list consists of full original features. Thirdly, full_new_features_list consists of full original features with new features. The newly generated features are constructed by existed features. There are two pairs of features, which are from_messages/from_this_person_to_poi and to_messages/from_poi_to_this_person. The ratio between poi-related emails count and total count could be reasonable features. So the new features are
1. from_poi_to_this_person_ratio = from_poi_to_this_person/to_messages
2. from_this_person_to_poi_ratio = from_this_person_to_poi/from_messages

The two new features are generated and added to the feature list.

And it's important to do feature scaling for those algorithms which needed, such as SVM, K-means.This will be done specifically in the features/labels splitting step, by the MinMaxScaler. 

### Algorithm tuning
In this section, 3 different algorithms are picked, Gaussian Naive Bayes, Decision Tree and Support Vector Machine. Each is deployed on the 3 different datasets. Firstly, the 3 algorithms are employed without any parameters tuning. Different algorithms are employed with 3 feature lists, the performances are listed as below.

|   Features Type   |  Algorithms  | Accuracy |  Precision  |  Recall |
|:-----------------:|:------------:|:--------:|:-----------:|:-------:|
|     Optimized     |  GaussianNB  |  0.82927 |   0.28373   | 0.18400 |
|        Full       |  GaussianNB  |  0.74713 |   0.23578   | 0.40000 |
| Full, New Feature |  GaussianNB  |  0.74713 |   0.23578   | 0.40000 |
|     Optimized     | DecisionTree |  0.82547 | **0.33973** | 0.32750 |
|        Full       | DecisionTree |  0.80073 |   0.23570   | 0.22050 |
| Full, New Feature | DecisionTree |  0.81620 |   0.30797   | 0.30350 |
|     Optimized     |   LinearSVC  |  0.64487 |   0.13544   | 0.30900 |
|        Full       |   LinearSVC  |  0.73727 |   0.20475   | 0.33650 |
| Full, New Feature |   LinearSVC  |  0.73727 |   0.20475   | 0.33650 |

From the table, the decision tree on the optimized features has the best precision, 0.33973, in the meantime, the recall reaches 0.32750. Compared to the original full features, the optimized features have a better precision performance, which is important in this case. Notice that the performance of the SVM grows better after feature scaling. Interestingly, SVM classifier under optimized features has a poorer performance, compared to that under the full features. 

Furthermore, these algorithm should be tuned in the next step. Since the optimized feature list has a better precision along with the fine recall, we choose to run the tuning on it. And there isn't much we can do with the Naive Bayes algorithm, so it's skipped the tuning step. We mainly focus on the Decision Tree and Support Vector Machine algorithm.

GridSearchCV function is utilized to tune the decision tree and support vector machine.

For the decision tree, the min_samples_split parameter is tuned, which is the minimum number of samples required to split an internal node, affecting the toughness of the fitting curve. The default value is 2, we set a range from 2 to 10, after the grid search, we set the best parameter to the classifier, and rerun the tester function. It turns out that the best parameter should be around 4. The accuracy raises from 0.82687 to 0.83047, the precision is also better than that without tuning. And if we change the parameter to a higher value, the performances will go down because of overfit problem.

For the support vector machine, the C parameter is picked for tuning, with a range from 10 to 100. C is the penalty parameter of the error term, larger C value means a more complex decision boundary. After the grid search, the accuracy raises, also, the precision is better. However, the overall performance of the SVM is poorer than that of the decision tree, so decision tree is chosen as the final classifier. 

All the results are listed in the under table.

| Algorithms on the optimized features | Accuracy | Precision |  Recall |
|:------------------------------------:|:--------:|:---------:|:-------:|
|   Naive Bayes (GaussianNB, untuned)  |  0.82927 |  0.28373  | 0.18400 |
|      Decision Tree (untuned)         |  0.82547 |  0.33973  | 0.33350 |
|       Decision Tree (tuned)          |  0.83453 |  **0.36758**  | 0.33450 |
|      SVM (LinearSVC, untuned)        |  0.64487 |  0.13544  | 0.30900 |
|       SVM (LinearSVC, tuned)         |  0.64547 |  0.13618  | 0.31050 |

### Validation and Evaluation

**Why validation is important?** In the section above, several algorithms are tuned for better performance. The goal of cross validation is to define a dataset to test the model in the training phase. Without it, the overfit problem could not be aware. So, all the features and labels are split into the training set and test set for validation. The cross validation could help to reduce the impact of the overfit problem, thus, we could get a trustworthy accuracy number. Furthermore, the POI ratio of the dataset is as low as 12.3%, in such cases using stratified sampling as implemented in StratifiedShuffleSplit to ensure that relative class frequencies is approximately preserved in each train and validation fold.

**Why evaluation metrics?** Also, during the tuning process, accuracy, precision and recall are employed as the benchmarks. With these benchmarks, the performance of the algorithms could be precisely qualified. Precision is the ratio of true positives and the sum of true positives and false positives, while recall is the ratio of true positives and the sum of true positives and false negatives. The definition equations are as below.
- Precision = true positives / (true positives + false positives)
- Recall = true positives / (true positives + false negatives)

Usually there's a trade-off between precision and recall, we could see that while tuning, the precision goes up with a lower recall. In this case, we want to have a precise poi identifier, so the slightly low recall could be accepted.

### Conclusion

In this project, poi exploration is done with the enron dataset. Firstly, we take a look at the basic information of the dataset. Secondly, we go deeper and visualize the data so the outliers are removed. Then, two new features are generated to present the poi/overall mail ratio. After that, 3 different algorithms are selected to perform the train and test. Untuned and tuned algorithms' performances are compared, with the grid search function, decision tree and svm show better accuracy and precision. And it turns out the decision tree algorithm has the best accuracy and precision. 

### References
1. Project rubics, https://review.udacity.com/#!/rubrics/27/view
2. Built-in Types, Python 2.7.14, https://docs.python.org/2/library/stdtypes.html
3. Making a flat list out of list of lists in Python, https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
4. Support Vector Machine, http://scikit-learn.org/stable/modules/svm.html#svm-classification
5. Naive Bayes, http://scikit-learn.org/stable/modules/naive_bayes.html
6. Decision Tree, http://scikit-learn.org/stable/modules/tree.html
