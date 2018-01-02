#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
#print(data_dict)
data_dict.pop("TOTAL")
for data in data_dict:
    #print(data, "\n", data_dict[data], "\n", data_dict[data]["bonus"])
    if data_dict[data]["bonus"] == "NaN" or data_dict[data]["salary"] == "NaN":
        continue
    elif data_dict[data]["bonus"] > 5000000 and data_dict[data]["salary"] > 1000000:
        print(data)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
for point in data:

    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


