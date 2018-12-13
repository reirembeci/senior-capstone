from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import sys
import random
import re
import helpers

# Returns all training data as a list which contains only the text(removing the source, paragraph #, sentence #, ratings
def get_list_re(filename):
    list = []
    file = open(filename, "r", encoding = "utf-8")
    for line in file.readlines():
        if line[0] != '[' and line != "\n":
            l = line.split("]\",")[-1]
            final = re.sub("[^a-zA-Z]"," ", l)
            list.append(final)

    return list

# preprocess the data so it can be used by the classifiers
def preprocess(samples, percent_test):
    num_samples = len(samples)
    random.seed(1234)
    random.shuffle(samples)
    cutoff = int((1.0 - percent_test) * num_samples)
    # create a train set and a test/development set
    feature_sets = [(text, label) for (text, label) in samples]
    train_set =  feature_sets[:cutoff]
    test_set = feature_sets[cutoff:]
    # separate the training data and the training labels
    train_data = [text for (text, label) in train_set]
    train_labels = [label for (text, label) in train_set]
    # separate the test data and the test labels
    test_data = [text for (text, label) in test_set]
    test_labels = [label for (text, label) in test_set]
    return(train_data, train_labels, test_data, test_labels)

# Transform the data so it can be represented using tfidf
def tfidf(train_data, test_data, extra):
    TfidfVect = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False, stop_words=extra['stop_words'], max_df=extra['max_df'], norm=extra['norm'], min_df=extra['min_df'])
    TfidfTrans = TfidfVect.fit_transform(train_data)
    TfidfTrans_test = TfidfVect.transform(test_data)
    return(TfidfTrans, TfidfTrans_test)

# Transform the data so it can be represented using Count Vectorizer
def countvect(train_data, test_data, extra):
    CountVect = CountVectorizer(lowercase=False, stop_words=extra['stop_words'], max_df=extra['max_df'])
    CountTrans = CountVect.fit_transform(train_data)
    CountTest = CountVect.transform(test_data)
    return(CountTrans, CountTest)

# Transform the data so it can be represented using Hashing Vectorizer
def hashing(train_data, test_data,extra, classifier=[]):
    if classifier == "naive":
        HashVect = HashingVectorizer(lowercase=False, non_negative=True, stop_words=extra['stop_words'],norm=extra['norm'])
    else:
         HashVect = HashingVectorizer(lowercase=False, stop_words=extra['stop_words'], norm=extra['norm'])
    HashTrans = HashVect.fit_transform(train_data)
    HashTest = HashVect.transform(test_data)
    return(HashTrans, HashTest)

# Implementetion of the fmeasure metric, which calculates the precision, recall and f1measure given a confusion matrix
def fmeasure(matrix):
    value1 = (matrix[0][1] + matrix[0][0])
    value2 = (matrix[1][0] + matrix[0][0])
    if value1 == 0 or value2 == 0:
        precision = 0
        recall = 0
        f_measure = 0
    else:
        precision = matrix[0][0] / value2
        recall = matrix[0][0] / value1
        if precision == 0:
            f_measure = 0
        else:
            f_measure = (2 * precision * recall) / (precision + recall)
    return(precision, recall, f_measure)


def get_representation(train_data, test_data, representation, classifier, extra):
    if representation == "tfidf":
        return tfidf(train_data, test_data, extra)
    elif representation == "count":
        return countvect(train_data, test_data, extra)
    elif representation == "hash":
        if classifier == "naive":
            return hashing(train_data, test_data, extra, "naive")
        else:
            return hashing(train_data, test_data, extra)
    else:
        sys.exit("This representation has not been implemented yet.")
        return None




