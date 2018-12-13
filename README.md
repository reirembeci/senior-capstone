# senior-capstone
Detecting textual analogies using semi-supervised learning

A system for detecting analogies in a given text using two semi supervised learning techniques; transductive support vector machines (TSVMs) and label propagation. Count vectorization, tf-idf, and hash vectorization are the explored feature extraction tools.

The following scripts are used to extract the corpora, build the training set and the testing set, and analyze the results.

a) compile_folder

compile_folder is used to extract and convert all the text files of the corpus to a CSV file. 

b) wordhunt 

wordhunt goes through each sentence in the CSV file generated by compile_folder and looks for phrases shown in that might suggest the presence of an analogy. It then creates two CSV files, one with sentences that include the aforementioned phrases, and one with sentences that don’t.

c) functions

functions build the training and the testing set, as well as extract the features from these sets

d) main_grid

main_grid implements the exhaustive search on the parameters. It takes as input the name of the classifier and the set of parameter values to be searched over. It returns the set of parameters which produced the highest score when training the classifier, along with the score.

e) main_interface

main_interface is the central script. It takes as input the positive set, the negative set, and name of classifier. Its output is the overall accuracy, precision, recall, f1-score, and the confusion matrix.

f) overlap-test

overlap-test runs an overlapping test in error between two sets of (classifier - feature extraction tool) pairs.
