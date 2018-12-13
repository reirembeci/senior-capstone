from __future__ import print_function
import functions
import helpers
#import parameters_file
#--------
#from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 
np.seterr(divide='ignore', invalid='ignore')

def main():
    positive_set = '../latest_analogy/test_extractions/bc_samples.txt' #'test_extractions/test-neural-hash-samples.txt' 
    negative_set = '../latest_analogy/test_extractions/bc_grounds.txt' #'test_extractions/test-neural-hash-ground.txt' 
    analogy_list = functions.get_list_re(positive_set)
    non_analogy_list = functions.get_list_re(negative_set)
    samples = [(text, 1) for text in analogy_list] + [(text, 0) for text in non_analogy_list]
    train_data, train_labels, test_data, test_labels = functions.preprocess(samples, 0.15)
    overlap_input = [('LP','count'), ('TSVM', 'tfidf')]
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(train_labels)) < 0.7
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    train_labels[random_unlabeled_points] = -1
    train_data = np.array(train_data)
    prediction_second_input = []
    pipeline = []
    no_as_yes = [] # predictions with label NO classified with label YES
    yes_as_no = [] # predictions with label YES classified with label NO
    count = 0
    
    for element in overlap_input:
        pipeline = helpers.get_function(element[0])
        representation = helpers.get_function(element[1])
        parameters = helpers.get_parameters(element[0])
        train_set = representation.fit_transform(train_data).toarray()
        test_set = representation.transform(test_data).toarray()
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=10, error_score=-1)
        grid_search.fit(train_set, train_labels)
        if count == 0:
            prediction = grid_search.best_estimator_.predict(test_set)
            matrix = confusion_matrix(test_labels, prediction, labels = [1, 0])   
        else:
            prediction_second_input = grid_search.best_estimator_.predict(test_set)
            matrix = confusion_matrix(test_labels, prediction_second_input, labels = [1, 0])   
        count += 1
        print(matrix)
        
    for i in range(len(test_labels)):
        #print(test_labels[i], prediction[i], prediction_second_input[i])
        if (test_labels[i] != prediction[i]) and (prediction[i] == prediction_second_input[i]):
            if test_labels[i] == 0:
                no_as_yes.append(test_data[i])
            else:
                yes_as_no.append(test_data[i])
    
    print("Overlapping NO as YES:")
    l1 = len(no_as_yes)
    print("Number: ", l1)
    for i in range(l1):
        print(no_as_yes[i])
    print("Overlapping YES as NO:")
    l2 = len(yes_as_no)
    print("Number: ", l2)
    for i in range(l2):
        print(yes_as_no[i])
    
main()