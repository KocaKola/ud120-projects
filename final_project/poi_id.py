#!/usr/bin/python

import pickle
import sys

import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Open the README.md and auto-generate the README document along with the code.
# explore the data set
# remove outliers
# feature selection & scaling
# pick algorithm and tune the parameters
# validation
# Evaluation Metrics
# performance, precision and recall

with open("../README.md", "w") as readme:
    readme.write(
        "# README project ud120\n\n"
        "### Introduction\n"
        "This README.md file includes those information which needed to explain. The "
        "process of the machine learning is a sequential combination of several operations, such as data cleaning, "
        "feature generation and scaling, parameters tuning and validation.\n\n")

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                 'director_fees']  # You will need to use more features
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'shared_receipt_with_poi']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Find out how many missing values for each feature of the data point
nan_finances = [0 for i in range(len(features_list))]
nan_emails = [0 for i in range(len(email_features))]
for i, f in enumerate(features_list):
    for _, v in enumerate(data_dict):
        nan_finances[i] += (data_dict[v][f] == "NaN")

for i, f in enumerate(email_features):
    for _, v in enumerate(data_dict):
        nan_emails[i] += (data_dict[v][f] == "NaN")

print "The NaN count of each feature in the feature_list is", nan_finances
print "The NaN count of each feature in the email_features is", nan_emails

# Remove those features whose half is missing
features_list_half = [features_list[i] for i, v in enumerate(nan_finances) if v < len(data_dict) // 2]

# Also remove the total value.
features_list_half.remove("total_payments")
features_list_half.remove("total_stock_value")
# print(str.join(",", features_list))

# Remove the email address which is string type
email_features.remove("email_address")

# Find out the number of POI and non-POI
num_poi = 0
for k, v in enumerate(data_dict):
    if data_dict[v]["poi"] == 1:
        num_poi += 1
print "The POI number is", num_poi

# Write down several findings about the data set, including the total number, features, etc.
with open("../README.md", "a") as readme:
    s = str.join(" ", ["### Data Exploration\n", "The total number of the data points is", str(len(data_dict)),
                       "and there're", str(len(data_dict.values()[0])), "features in each point.",
                       "Among the features, there're two types, finances and emails,",
                       "separated as features_list and email_features in the code.",
                       "Those features which missed less than half of the data size are remained",
                       "since too much NaN could lead to the overfit problem.",
                       "Besides, two total value features are dropped, then the left features are",
                       str.join(", ", features_list), "where the poi label included.",
                       "Also, the email features are remained except the email_address string.\n\n",

                       "The numbers of the POI and non-POI are", str(num_poi), "and", str(len(data_dict) - num_poi),
                       "respectively. The ratio of POI is about",
                       str("{:.2%}.".format(float(num_poi) / len(data_dict))),
                       "The numbers of the NaN existed in each features are described as below.\n\n",
                       "|        finance feature    | NaN counts |      email feature      | NaN counts |\n"
                       "|:-------------------------:|:----------:|:-----------------------:|:----------:|\n"
                       "|            poi            |      0     |       to_messages       |     60     |\n"
                       "|           salary          |     51     |      email_address      |     35     |\n"
                       "|     deferral_payments     |     107    | from_poi_to_this_person |     60     |\n"
                       "|       total_payments      |     21     |      from_messages      |     60     |\n"
                       "|       loan_advances       |     142    | from_this_person_to_poi |     60     |\n"
                       "|           bonus           |     64     | shared_receipt_with_poi |     60     |\n"
                       "| restricted_stock_deferred |     128    |                         |            |\n"
                       "|      deferred_income      |     97     |                         |            |\n"
                       "|     total_stock_value     |     20     |                         |            |\n"
                       "|          expense          |     51     |                         |            |\n"
                       "|  exercised_stock_options  |     44     |                         |            |\n"
                       "|           other           |     53     |                         |            |\n"
                       "|    long_term_incentive    |     80     |                         |            |\n"
                       "|      restricted_stock     |     36     |                         |            |\n"

                       "\n\n"])
    readme.write(s)

# Task 2: Remove outliers
# Start with the first 2 features, salary and bonus.
# Visualization is utilized to find the outliers.
data = featureFormat(data_dict, features_list_half)
for point in data:
    salary = point[1]
    bonus = point[2]
    plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
# plt.show()

# It turns out there exists a extremely large point called "TOTAL".
data_dict.pop("TOTAL")
data = featureFormat(data_dict, features_list_half)

# Then the visualization again. This time we go through all the features using salary as X-axis.
for i, f in enumerate(features_list_half[2:]):
    for point in data:
        salary = point[1]
        feature = point[i + 2]
        plt.scatter(salary, feature)
    plt.xlabel("salary")
    plt.ylabel(f)
    # plt.show()

with open("../README.md", "a") as readme:
    readme.write(
        "### Deal with the Outlier\n"
        "We start with the salary and bonus feature, "
        "and it turns out there exists a TOTAL data point which add up all. "
        "After cleaning the TOTAL outlier, we go through all the 5 features compared to the salary. "
        "It still seems that there're probably 4 more outliers in the dataset. "
        "However, these outliers are reasonable data points, it shouldn't be removed.\n\n")

# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
# my_dataset: cut off the features have more than half NaN, adding new features
my_dataset = data_dict
# full_feature_dataset: full original features, no new features
full_feature_dataset = data_dict
# full_new_feature_dataset: full original features, adding new features
full_new_feature_dataset = data_dict

with open("../README.md", "a") as readme:
    readme.write("### Optimize the Features\n"
                 "Firstly, I thought the email features could be used to generate new features. "
                 "I've rewrite the getToFromStrings function in the poi_flag_email.py "
                 "to create the from/to poi features, "
                 "but it turns out that they were provided in the original data dict. :(\n\n"
                 )

# sys.path.append("../feature_selection/")
# from poi_flag_email import getToFromStrings
# from poi_email_addresses import poiEmails
# import os
# for from_who in os.listdir("emails_by_address"):
#     if from_who == ".DS_Store":
#         continue
#     from_who = os.path.join("emails_by_address/", from_who)
#     with open(from_who, "r") as email_path:
#         for path in email_path:
#             path = str.join("/", path.split("/")[1:])
#             path = os.path.join("../", path[:-1])
#             with open(path, "r") as message:
#                 to_addresses, from_addresses, cc_addresses = getToFromStrings(message.read())

with open("../README.md", "a") as readme:
    readme.write("So I think, the relations between the features in the original dataset "
                 "could be used as new features. Thus, 3 set of features list are generated for comparison. "
                 "Firstly, my_feature_list consists of cut-off features list with newly generated features. "
                 "Secondly, full_features_list consists of full original features. "
                 "Thirdly, full_new_features_list consists of full original features with new features. "
                 "The newly generated features are constructed by existed features. "
                 "There are two pairs of features, which are from_messages/from_this_person_to_poi "
                 "and to_messages/from_poi_to_this_person. The ratio between poi-related emails count and total count "
                 "could be reasonable features. So the new features are\n"
                 "1. from_poi_to_this_person_ratio = from_poi_to_this_person/to_messages\n"
                 "2. from_this_person_to_poi_ratio = from_this_person_to_poi/from_messages\n\n"
                 "The two new features are generated and added to the feature list.\n\n"
                 "And it's important to do feature scaling for those algorithms which needed, such as SVM, K-means."
                 "This will be done specifically in the features/labels splitting step, by the MinMaxScaler. "
                 "\n\n"
                 )

# Generate the two new float features.
for i in my_dataset:
    if my_dataset[i]["from_poi_to_this_person"] != "NaN" and my_dataset[i]["to_messages"] != "NaN":
        my_dataset[i]["from_poi_to_this_person_ratio"] = float(my_dataset[i]["from_poi_to_this_person"]) / \
                                                         my_dataset[i]["to_messages"]
    else:
        my_dataset[i]["from_poi_to_this_person_ratio"] = "NaN"

    if my_dataset[i]["from_this_person_to_poi"] != "NaN" and my_dataset[i]["from_messages"] != "NaN":
        my_dataset[i]["from_this_person_to_poi_ratio"] = float(my_dataset[i]["from_this_person_to_poi"]) / \
                                                         my_dataset[i]["from_messages"]
    else:
        my_dataset[i]["from_this_person_to_poi_ratio"] = "NaN"

# Renew the features list
my_features_list = features_list_half + email_features + ["from_poi_to_this_person_ratio",
                                                          "from_this_person_to_poi_ratio"]
full_features_list = features_list + email_features
full_new_features_list = features_list + email_features + ["from_poi_to_this_person_ratio",
                                                           "from_this_person_to_poi_ratio"]

# Extract features and labels from dataset for local testing
my_data = featureFormat(my_dataset, my_features_list, sort_keys=True)
full_data = featureFormat(data_dict, full_features_list, sort_keys=True)
full_new_data = featureFormat(my_dataset, full_new_features_list, sort_keys=True)

labels, features = targetFeatureSplit(my_data)
full_labels, full_features = targetFeatureSplit(full_data)
full_new_labels, full_new_features = targetFeatureSplit(full_new_data)

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)
full_features = min_max_scaler.fit_transform(full_features)
full_new_features = min_max_scaler.fit_transform(full_new_features)

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.naive_bayes import GaussianNB

clfNB = GaussianNB()
clfNB_full = GaussianNB()
clfNB_full_new = GaussianNB()

from sklearn.tree import DecisionTreeClassifier

clfDT = DecisionTreeClassifier(random_state=42)
clfDT_full = DecisionTreeClassifier(random_state=42)
clfDT_full_new = DecisionTreeClassifier(random_state=42)

from sklearn import svm

clfSVM = svm.SVC(random_state=42)
clfSVM_full = svm.SVC(random_state=42)
clfSVM_full_new = svm.SVC(random_state=42)
# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
full_features_train, full_features_test, full_labels_train, full_labels_test = \
    train_test_split(full_features, full_labels, test_size=0.3, random_state=42)
full_new_features_train, full_new_features_test, full_new_labels_train, full_new_labels_test = \
    train_test_split(full_new_features, full_new_labels, test_size=0.3, random_state=42)

# Import the test_classifier function in tester to get the accuracy of each algorithm.
from tester import test_classifier

# NaiveBayes without tuning
clfNB.fit(features_train, labels_train)
print("NB without tuning on the optimized dataset")
test_classifier(clfNB, my_dataset, my_features_list, folds=1000)

clfNB_full.fit(full_features_train, full_labels_train)
print("NB without tuning on the full original dataset")
test_classifier(clfNB_full, data_dict, full_features_list, folds=1000)

clfNB_full_new.fit(full_new_features_train, full_new_labels_train)
print("NB without tuning on the full dataset added up new features")
test_classifier(clfNB_full_new, my_dataset, full_new_features_list, folds=1000)

# DecisionTree without tuning
clfDT.fit(features_train, labels_train)
print("DT without tuning on the optimized dataset")
test_classifier(clfDT, my_dataset, my_features_list, folds=1000)

clfDT_full.fit(features_train, labels_train)
print("DT without tuning on the full original dataset")
test_classifier(clfDT_full, data_dict, full_features_list, folds=1000)

clfDT_full_new.fit(features_train, labels_train)
print("DT without tuning on the full dataset added up new features")
test_classifier(clfDT_full_new, my_dataset, full_new_features_list, folds=1000)

# SVM without tuning
clfSVM.fit(features_train, labels_train)
print("SVM without tuning on the optimized dataset")
test_classifier(clfSVM, my_dataset, my_features_list, folds=1000)

clfSVM_full.fit(full_features_train, full_labels_train)
print("SVM without tuning on the full original dataset")
test_classifier(clfSVM_full, data_dict, full_features_list, folds=1000)

clfSVM_full_new.fit(full_new_features_train, full_new_labels_train)
print("SVM without tuning on the full dataset added up new features")
test_classifier(clfSVM_full_new, my_dataset, full_new_features_list, folds=1000)

# SVM without tuning and the LinearSVC()
clfSVM = svm.LinearSVC(random_state=42)
clfSVM.fit(features_train, labels_train)
print("SVM without tuning, changed to LinearSVC() on the optimized dataset")
test_classifier(clfSVM, my_dataset, my_features_list, folds=1000)

clfSVM_full = svm.LinearSVC(random_state=42)
clfSVM_full.fit(full_features_train, full_labels_train)
print("SVM without tuning, changed to LinearSVC() on the full original dataset")
test_classifier(clfSVM_full, data_dict, full_features_list, folds=1000)

clfSVM_full_new = svm.LinearSVC(random_state=42)
clfSVM_full_new.fit(full_new_features_train, full_new_labels_train)
print("SVM without tuning, changed to LinearSVC() on the full dataset added up new features")
test_classifier(clfSVM_full_new, my_dataset, full_new_features_list, folds=1000)

with open("../README.md", "a") as readme:
    readme.write("### Algorithm tuning\n"
                 "In this section, 3 different algorithms are picked, Gaussian Naive Bayes, Decision Tree and Support "
                 "Vector Machine. Each is deployed on the 3 different datasets. "
                 "Firstly, the 3 algorithms are employed without any parameters tuning. "
                 "Different algorithms are employed with 3 feature lists, the performances are listed as below.\n\n"
                 "|   Features Type   |  Algorithms  | Accuracy |  Precision  |  Recall |\n"
                 "|:-----------------:|:------------:|:--------:|:-----------:|:-------:|\n"
                 "|     Optimized     |  GaussianNB  |  0.82927 |   0.28373   | 0.18400 |\n"
                 "|        Full       |  GaussianNB  |  0.74713 |   0.23578   | 0.40000 |\n"
                 "| Full, New Feature |  GaussianNB  |  0.74713 |   0.23578   | 0.40000 |\n"
                 "|     Optimized     | DecisionTree |  0.82547 | **0.33973** | 0.32750 |\n"
                 "|        Full       | DecisionTree |  0.80073 |   0.23570   | 0.22050 |\n"
                 "| Full, New Feature | DecisionTree |  0.81620 |   0.30797   | 0.30350 |\n"
                 "|     Optimized     |   LinearSVC  |  0.64487 |   0.13544   | 0.30900 |\n"
                 "|        Full       |   LinearSVC  |  0.73727 |   0.20475   | 0.33650 |\n"
                 "| Full, New Feature |   LinearSVC  |  0.73727 |   0.20475   | 0.33650 |\n\n"
                 "From the table, the decision tree on the optimized features has the best precision, 0.33973, in the "
                 "meantime, the recall reaches 0.32750. Compared to the original full features, the optimized features "
                 "have a better precision performance, which is important in this case. Notice that the performance "
                 "of the SVM grows better after feature scaling. Interestingly, SVM classifier under optimized "
                 "features has a poorer performance, compared to that under the full features. \n\n"

                 "Furthermore, these algorithm should be tuned in the next step. Since the optimized feature list has "
                 "a better precision along with the fine recall, we choose to run the tuning on it. "
                 "And there isn't much we can do with the Naive Bayes algorithm, so it's skipped the tuning step. "
                 "We mainly focus on the Decision Tree and Support Vector Machine algorithm.\n\n"
                 )

# DT with GridSearchCV tuning
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

cv = cross_validation.StratifiedShuffleSplit(labels_train, random_state=42)

parameters = {"min_samples_split": [x for x in range(2, 20, 2)]}
clfDT = DecisionTreeClassifier(random_state=42)
clfDT_CV = GridSearchCV(clfDT, parameters, cv=cv, scoring='precision')
clfDT_CV.fit(features_train, labels_train)
print("DT with GridSearchCV tuning")
print("Best parameters are,", clfDT_CV.best_params_)
clfDT = DecisionTreeClassifier(min_samples_split=clfDT_CV.best_params_["min_samples_split"], random_state=42)
clfDT.fit(features_train, labels_train)
test_classifier(clfDT, my_dataset, my_features_list, folds=1000)

with open("../README.md", "a") as readme:
    readme.write("GridSearchCV function is utilized to tune the decision tree and support vector machine.\n\n"
                 "For the decision tree, the min_samples_split parameter is tuned, which is "
                 "the minimum number of samples required to split an internal node, affecting the toughness of the "
                 "fitting curve. The default value is 2, we set a range from 2 to 10, after the grid search, "
                 "we set the best parameter to the classifier, and rerun the tester function. It turns "
                 "out that the best parameter should be around 4. The accuracy raises from 0.82687 to 0.83047, "
                 "the precision is also better than that without tuning. And if we change the parameter "
                 "to a higher value, the performances will go down because of overfit problem.\n\n"
                 )

# SVM with GridSearchCV tuning
parameters = {'C': [x for x in range(10, 100, 10)]}
clfSVM = svm.LinearSVC(random_state=42)
clfSVM_CV = GridSearchCV(clfSVM, parameters, cv=cv, scoring='precision')
clfSVM_CV.fit(features_train, labels_train)
print("SVM with GridSearchCV tuning")
print("Best parameters are,", clfSVM_CV.best_params_)
clfSVM = svm.LinearSVC(C=clfSVM_CV.best_params_["C"], random_state=42)
test_classifier(clfSVM, my_dataset, my_features_list, folds=1000)

with open("../README.md", "a") as readme:
    readme.write("For the support vector machine, the C parameter is picked for tuning, with a range from 10 to 100. "
                 "C is the penalty parameter of the error term, larger C value means a more complex decision boundary. "
                 "After the grid search, the accuracy raises, also, the "
                 "precision is better. However, the overall performance of the SVM is poorer than that "
                 "of the decision tree, so decision tree is chosen as the final classifier. \n\n"
                 "All the results are listed in the under table.\n\n"
                 "| Algorithms on the optimized features | Accuracy | Precision |  Recall |\n"
                 "|:------------------------------------:|:--------:|:---------:|:-------:|\n"
                 "|   Naive Bayes (GaussianNB, untuned)  |  0.82927 |  0.28373  | 0.18400 |\n"
                 "|      Decision Tree (untuned)         |  0.82547 |  0.33973  | 0.33350 |\n"
                 "|       Decision Tree (tuned)          |  0.83453 |  **0.36758**  | 0.33450 |\n"
                 "|      SVM (LinearSVC, untuned)        |  0.64487 |  0.13544  | 0.30900 |\n"
                 "|       SVM (LinearSVC, tuned)         |  0.64547 |  0.13618  | 0.31050 |\n\n"
                 )

with open("../README.md", "a") as readme:
    readme.write("### Validation and Evaluation\n\n"
                 "**Why validation is important?** In the section above, several algorithms are tuned for "
                 "better performance. The goal of cross validation is to define a dataset"
                 " to test the model in the training phase. Without it, the overfit problem could not be aware. "
                 "So, all the features and labels are split into the training set and test set for validation. "
                 "The cross validation could help to reduce the impact of the overfit problem, thus, we could "
                 "get a trustworthy accuracy number. Furthermore, the POI ratio of the dataset is as low as 12.3%, "
                 "in such cases using stratified sampling as implemented in StratifiedShuffleSplit to ensure that "
                 "relative class frequencies is approximately preserved in each train and validation fold.\n\n"

                 "**Why evaluation metrics?** Also, during the tuning process, accuracy, precision and recall are "
                 "employed as the benchmarks. "
                 "With these benchmarks, the performance of the algorithms could be precisely qualified. "
                 "Precision is the ratio of true positives and the sum of true positives and false positives, while "
                 "recall is the ratio of true positives and the sum of true positives and false negatives. "
                 "The definition equations are as below.\n"
                 "- Precision = true positives / (true positives + false positives)\n"
                 "- Recall = true positives / (true positives + false negatives)\n\n"
                 "Usually there's a trade-off between precision and recall, we could see that while tuning, "
                 "the precision goes up with a lower recall. In this case, we want to have a precise poi identifier, "
                 "so the slightly low recall could be accepted.\n\n"
                 )

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.
clf = clfDT
dump_classifier_and_data(clf, my_dataset, features_list)

with open("../README.md", "a") as readme:
    readme.write("### Conclusion\n\n"
                 "In this project, poi exploration is done with the enron dataset. Firstly, we take a look at "
                 "the basic information of the dataset. Secondly, we go deeper and visualize the data so the outliers "
                 "are removed. Then, two new features are generated to present the poi/overall mail ratio. "
                 "After that, 3 different algorithms are selected to perform the train and test. Untuned and tuned "
                 "algorithms' performances are compared, with the grid search function, decision tree and svm show "
                 "better accuracy and precision. And it turns out the decision tree algorithm has the best "
                 "accuracy and precision. \n\n")
    readme.write("### References\n"
                 "1. Project rubics, https://review.udacity.com/#!/rubrics/27/view\n"
                 "2. Built-in Types, Python 2.7.14, https://docs.python.org/2/library/stdtypes.html\n"
                 "3. Making a flat list out of list of lists in Python, "
                 "https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python\n"
                 "4. Support Vector Machine, http://scikit-learn.org/stable/modules/svm.html#svm-classification\n"
                 "5. Naive Bayes, http://scikit-learn.org/stable/modules/naive_bayes.html\n"
                 "6. Decision Tree, http://scikit-learn.org/stable/modules/tree.html\n"
                 )
