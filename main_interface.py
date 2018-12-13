from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import confusion_matrix
import numpy as np
import helpers
import functions
from sklearn.feature_extraction.text import TfidfVectorizer
from scikitTSVM import SKTSVM
import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 
 
tsvm = SKTSVM(probability=False,C=0.01,gamma=1.0,kernel= 'linear',lamU= 1.0)
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
TfidfVect = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
train_set = TfidfVect.fit_transform(train_data).toarray()
test_set = TfidfVect.transform(test_data).toarray()

# Label Propagation
"""
label_prop_model = helpers.get_function('LP')
label_prop_model.fit(train_set, train_labels)
test_predict = label_prop_model.predict(test_set)
print(label_prop_model.score(test_set, test_labels))
"""

print("Total size of training set: ",len(train_labels))
i = 0
for l in train_labels:
    if l==-1:
        i += 1
print("Size of unlabeled data: ", i)
print("Size of the testing set ",len(test_data))

# TSVM
#"""
tsvm.fit(train_set, train_labels)
test_predict = tsvm.predict(test_set)
print("Accuracy: ", tsvm.score(test_set, test_labels))
#"""
print("Confusion matrix:")
matrix = confusion_matrix(test_labels, test_predict, labels = [1, 0])
print(matrix)
precision, recall, f_measure = functions.fmeasure(matrix)
print("Precision: ", precision)
print("Recall: ", recall)
print("f_measure: ", f_measure)