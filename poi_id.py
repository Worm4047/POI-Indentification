#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")
import numpy as np
import matplotlib.pyplot
import os
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'shared_receipt_with_poi',
                 'expenses'
                 ]

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
data_dict.pop("TOTAL",0)
#Popping out the TOTAL ROW AS WAS OBSERVED IN OUTLIERS SECTION , IT IS ONE OF THE MAIN OUTLIER.


###Removing items with bonus greater than 6000000 ################# 
new_dict={}
for item in data_dict:
  bonus = data_dict[item]['bonus']
  salary = data_dict[item]['salary']
  if bonus > 6000000 :continue
  new_dict[item]=data_dict[item]
data_dict=new_dict  

###ADDING NEW FEATURES ##########

def fraction_to_this_person_email(data):
  for item in data:
    to_msg = data[item]['to_messages']
    from_poi_to_this_person=data[item]['from_poi_to_this_person']
    if to_msg == 'NaN' or from_poi_to_this_person == 'NaN':data[item]['frac_from_poi_to_this_person'] =0.0
    else:data[item]['frac_from_poi_to_this_person'] = float(from_poi_to_this_person)/int(to_msg)
  return data

def fraction_from_this_person_email(data):
  for item in data:
    from_msg = data[item]['from_messages']
    from_this_person_to_poi=data[item]['from_this_person_to_poi']
    if from_msg == 'NaN' or from_this_person_to_poi == 'NaN':data[item]['frac_from_this_person_to_poi'] =0.0
    else:data[item]['frac_from_this_person_to_poi'] = float(from_this_person_to_poi)/int(from_msg)
  return data

data_dict = fraction_to_this_person_email(data_dict)
data_dict = fraction_from_this_person_email(data_dict)
features_list.append('frac_from_this_person_to_poi')
features_list.append('frac_from_poi_to_this_person')
my_dataset = data_dict

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

###NAIVE BAYES######
"""Uncomment to try
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features, labels)
"""

###SVM #####
"""Uncomment to try """
from sklearn.svm import SVC
clf = SVC(C=1.0, kernel="rbf")
clf.fit(features, labels)

###DECISION TREE ###
"""from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=20, min_samples_split=20)
clf = clf.fit(features, labels)
"""

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)

"""uncomment to print the scatterplot 
for item in data_dict:
  salary = data_dict[item]['salary']
  bonus = data_dict[item]['bonus']
  matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
"""