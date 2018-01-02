#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print(len(enron_data.values()[0]))

num_poi = 0
for k, v in enumerate(enron_data):
    print(k,v)
    if enron_data[v]["poi"] == 1:
        num_poi += 1
print(num_poi)

print(enron_data["PRENTICE JAMES"])

print(enron_data["COLWELL WESLEY"])

print(enron_data["SKILLING JEFFREY K"])

print(enron_data["LAY KENNETH L"])

print(enron_data["FASTOW ANDREW S"])

num_salary = 0
num_email = 0
for k, v in enumerate(enron_data):
    if isinstance(enron_data[v]["salary"], int):
        num_salary +=1
    if enron_data[v]["email_address"] != "NaN":
        num_email +=1

print(num_salary)
print(num_email)