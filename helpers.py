import nltk
import sklearn.metrics
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.semi_supervised import LabelPropagation
from scikitTSVM import SKTSVM



def rbf_kernel_safe(X, Y=None, gamma=None): 
    X, Y = sklearn.metrics.pairwise.check_pairwise_arrays(X, Y) 
    if gamma is None: 
        gamma = 1.0 / X.shape[1] 

    K = sklearn.metrics.pairwise.euclidean_distances(X, Y, squared=True) 
    K *= -gamma 
    K -= K.max()
    np.exp(K, K)    # exponentiate K in-place 
    return K 

def get_function(name):
    if name == "LP":
        return(LabelPropagation(kernel=rbf_kernel_safe))
    elif name == "TSVM":
        return(SKTSVM(probability=False))
    elif name == "hash":
        return(HashingVectorizer())
    elif name == "count":
        return(CountVectorizer())
    elif name == "tfidf":
        return(TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False))
    
def get_parameters(name):
    if name == "LP":
        return(parameters_LP)
    elif name == "TSVM":
        return(parameters_TSVM)
    elif name == "hash":
        return(parameters_hash)
    elif name == "count":
        return(parameters_count)
    elif name == "tfidf":
        return(parameters_tfidf)

def merge_two_dicts(x, y):
    z = x.copy()   
    z.update(y)    
    return z

parameters_count = {
    'count__max_df': (0.5, 0.75, 0.8),
    'count__max_features': (None, 5000, 10000, 50000),
    'count__ngram_range': ((1, 1),(1, 2)),  # unigrams or bigrams
    #'count__strip_accents' : ('ascii', 'unicode', None),
    'count__analyzer' : ('word', 'char', 'char_wb'),    
    #'count__stop_words' : ('english', None),
    'count__min_df': (0.1, 0.2, 0.3)                        
}
    
parameters_tfidf = {
    'tfidf__max_df': (0.5, 0.75, 0.8),
    'tfidf__max_features': (None, 5000, 10000, 50000),
    'tfidf__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__strip_accents' : ('ascii', 'unicode', None),
    'tfidf__analyzer' : ('word', 'char', 'char_wb'),    
    #'tfidf__stop_words' : ('english', None),
    #'tfidf__min_df': (0.1, 0.2, 0.3, 0.4),   
    #'tfidf__norm' : ('l1', 'l2', None),
    'tfidf__use_idf': (True, False)
}

parameters_hash = {
    'hash__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'hash__strip_accents' : ('ascii', 'unicode', None),
    'hash__analyzer' : ('word', 'char', 'char_wb'),    
    #'hash__stop_words' : ('english', None),
    'hash__norm' : ('l1', 'l2', None)
}

parameters_LP = {
    'n_neighbors' : (2,5,7,13,17),
    'tol' : (0.001, 0.004, 0.01),
    'max_iter' : (1000, 2000)
}

parameters_TSVM = {
    'C' : (0.0001, 0.0004, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0),
    'kernel' : ('linear', 'RBF'),
    'gamma' : (1.0, 2.0),
    'lamU' : (1.0, 1.5, 2.0)
}

def generate_parameters(representation, classifier):
    part1 = get_parameters(representation)
    part2 = get_parameters(classifier)
    parameters = merge_two_dicts(part1, part2)
    return(parameters)