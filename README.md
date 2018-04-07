# Udacity Introduction to machine learning

## 1. Dataset and goal of project

## Goal 
The main purpose of project is develop the machine learning algorithm to detect person of interest(POI) from dataset.
Who is a POI ?
-> A person who has been in any way linked with the fraud or played an active participation in it.

## Dataset used 
We have Enron email+financial (E+F) dataset. 

There's example of one POI data point: 

	[{"SKILLING JEFFREY K":{
				'salary': 1111258, 
				'to_messages': 3627, 
				'deferral_payments': 'NaN', 
				'total_payments': 8682716, 
				'exercised_stock_options': 19250000, 
				'bonus': 5600000, 
				'restricted_stock': 6843672, 
				'shared_receipt_with_poi': 2042, 
				'restricted_stock_deferred': 'NaN', 
				'total_stock_value': 26093672, 
				'expenses': 29336, 
				'loan_advances': 'NaN', 
				'from_messages': 108, 
				'other': 22122, 
				'from_this_person_to_poi': 30, 
				'poi': True, 
				'director_fees': 'NaN', 
				'deferred_income': 'NaN', 
				'long_term_incentive': 1920000, 
				'email_address': 'jeff.skilling@enron.com', 
				'from_poi_to_this_person': 88
				}
	}]

## Outliers
Dataset contains some outliers. 

1) The 'TOTAL' row is the biggest Enron E+F dataset outlier. We should remove it from dataset.
	(As we have also already seen in outliers lesson os the class).
Moreover, there are  2 more outliers with bonus greater than 6000000 , they are removed and it results in increase in performance.

## 2. Feature selection process
I selected the following features : ['poi','salary','shared_receipt_with_poi','expenses']
I started with basic two features 'poi' and 'salary' that were provided and one by one started adding each feature
and checked results . I finally got satisfactory results with these features.

## New features
In addition I create two new features which were considered in course:
* `frac_from_poi_to_this_person` fraction of messages to that person from a POI
* `frac_from_this_person_to_poi` fraction of messages from that person to a POI

A person who is suspected to be a POI will surely have more communication via email with a POI
 
To verify my intuition i checked and compared my results using Decision Tree and i was satisied by the results.

## 3. Pick an algorithm
I tried the Naive Bayes, SVM and Decision Trees algorithms. 

## All results of examination I included in the following table


		**Naive Bayes**		Accuracy: 0.70388       Precision: 0.00536      Recall: 0.00100 F1: 0.00169     F2: 0.00119
		**Decision Trees**	 Accuracy: 0.80163       Precision: 0.58494      Recall: 0.71100 F1: 0.64184     F2: 0.68162

		**SVM**				ERROR !!!!

SVM algorithm returned the next error :
```
Got a divide by zero when trying out: SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
```

## Chosen algorithm
Based on best performance level I picked Decision Trees as a final algorithm.

## 4. Tune the algorithm
## Reasons for algorithm tuning
The main reason is get better results from algorithm. Parameters of ML classifiers have a big influence in output results. 
The purpose of tuning is to find best sets of parameters for particular dataset.

## GridSearchCV
I apply GridSearchCV to tune the following parameters

	|Parameter          	Settings for investigation
	|min_samples_split	 	[1-60]                    
	|random_state	     	[1-50]                    


As a result i obtained best performance when random_state=15, min_samples_split=3
but when i applied random_states=20 and min_sample_split=20 i got even better performance so i used those settings . 

## 5. Validation
Classic rookie mistake that anyone can do is using the same set for testing that we used for training.

To validate my analysis I used stratified shuffle split cross validation StratifiedShuffleSplit developed by Udacity and defined in tester.py file

## 6. Evaluation metrics
For evaluation purpose i used : Precision,recall,accuracy,true positives,false positives,false negatives,true negatives .
Final results can be found in table below

	|**metric**				**value**
	|**Precision**			**0.58494**
	|**Recall**				**0.71100**
	|**Accuracy** 			**0.80163**
	|**True positives** 	**1422**
	|**False positives**	**1009**
	|**False negatives**	**578**
	|**True negatives**		**4991**

	
## Conclusion
Precision and Recall  both higher than .3. Thus, project goal was reached.
Precision 0.58494 means when model detect person as POI it was true only in 58.5% cases. 
At the same time Recall 0.71100 says only 71.1% of all POIs was detected.

We have very imbalanced classes in E+F dataset. In addition, almost half of all POIs weren't included in dataset. 
In such conditions result we received good enough, but it's not perfect, of course.

# Algorithm outputs log

## features_list: 
	features_list = ['poi',
                 	'salary',
                 	'shared_receipt_with_poi',
                 	'expenses',
                 	'frac_from_poi_to_this_person',
                 	'frac_from_this_person_to_poi'
                 ]
				 
## Classifier:
	clf = tree.DecisionTreeClassifier()

## Metrics:
	 Accuracy: 0.80163       Precision: 0.58494      Recall: 0.71100 F1: 0.64184     F2: 0.68162
     Total predictions: 8000 True positives: 1422    False positives: 1009   False negatives:  578 
     True negatives: 4991

## Classifier:
	clf = GaussianNB()

## Metrics:
	  Accuracy: 0.70388       Precision: 0.00536      Recall: 0.00100 F1: 0.00169     F2: 0.00119
      Total predictions: 8000 True positives:    2    False positives:  371   False negatives: 1998   True negatives: 5629

## Tune the algorithm
## features_list
	features_list = ['poi',
                 	'salary',
                 	'shared_receipt_with_poi',
                 	'expenses',
                 	'frac_from_poi_to_this_person',
                 	'frac_from_this_person_to_poi'
                 ]
## best estimator:
	DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=20, min_weight_fraction_leaf=0.0,
            presort=False, random_state=20, splitter='best')


# Related links
	- [Documentation of scikit-learn 0.15][1]
	- [Selecting good features � Part I: univariate selection][2]
	- [Cross-validation: the right and the wrong way][3]


		[1]: http://scikit-learn.org/stable/documentation.html
	
		[2]: http://blog.datadive.net/selecting-good-features-part-i-univariate-selection/
	
		[3]: http://scikit-learn.org/stable/modules/cross_validation.html 
