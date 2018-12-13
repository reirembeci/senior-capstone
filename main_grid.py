from __future__ import print_function
import functions
import helpers
import random
import sys
#--------
import timeout 
import math
from pprint import pprint
from time import time
import logging
#--------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from scikitTSVM import SKTSVM
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def accuracy_from_matrix(matrix):
    total = matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1]
    correct = matrix[0][0] + matrix[1][1]
    accuracy = correct / total
    return(accuracy)

def main():
    percent_test = 0.15
    positive_set = 'data/bc_samples.txt'
    negative_set = 'data/bc_grounds.txt'
    unlabeled_set = 'data/unlabeled-data.csv'
    analogy_list = functions.get_list_re(positive_set)
    non_analogy_list = functions.get_list_re(negative_set)
    unlabeled_list = functions.get_list_re(unlabeled_set)
    samples = [(text, 1) for text in analogy_list] + [(text, 0) for text in non_analogy_list] 
    train_data, train_labels, test_data, test_labels = functions.preprocess(samples, percent_test)
    j = 0
    for sample in unlabeled_list:
        if j <= 20000:
            train_data.append(sample)
            train_labels.append(-1)
        j += 1
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    train_data = np.array(train_data)
    classifiers = ['TSVM']
    representations = ['tfidf','count','hash']
    
    for classifier in classifiers:
        for representation in representations:
            pipeline = helpers.get_function(classifier)
            print(pipeline)
            print(representation)
            rep = helpers.get_function(representation)
            train_set = rep.fit_transform(train_data).toarray()
            test_set = rep.transform(test_data).toarray()
            parameters = helpers.get_parameters(classifier)
            grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, error_score=-1)
            print("Performing grid search...")
            #print("pipeline:", [name for name, _ in pipeline.steps])
            print("parameters:")
            pprint(parameters)
            t0 = time()            
            grid_search.fit(train_set, train_labels)
            print("done in %0.3fs" % (time() - t0))
            print()
            
            print("Best score: %0.3f" % grid_search.best_score_)
            print("Best parameters set:")
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
            print()
            
            print("Getting the confusion matrix for the best estimator:")
            prediction = grid_search.best_estimator_.predict(test_set)
            matrix = confusion_matrix(test_labels, prediction, labels = [1, 0])
            precision, recall, f_measure = functions.fmeasure(matrix)
            accuracy = accuracy_from_matrix(matrix)
            print("Accuracy ", accuracy)
            print("Precision, recall, f-score:")
            print(precision, recall, f_measure)
            print(matrix)
            print()
            
main()