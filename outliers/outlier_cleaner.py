#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import math
    cleaned_data = []

    ### your code goes here

    cleaned_data = [(ages[i], net_worths[i], predictions[i] - net_worths[i]) for i in range(len(ages))]
    cleaned_data = sorted(cleaned_data, key = lambda i: math.fabs(i[-1]))
    ten = len(cleaned_data)//10
    print(ten)
    cleaned_data = cleaned_data[:len(cleaned_data) - ten]
    print(len(cleaned_data))
    return cleaned_data

