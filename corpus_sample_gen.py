import random

filename = 'oanc_pos.csv'
newfilename = 'oanc_sample_pos.csv'

file = open(filename, "r", encoding = "utf-8")
corpus = file.readlines()
length = len(corpus)
print(length)
"""
list_of_indices = random.sample(range(1, length), 300)
newfile = open(newfilename, "w")
for index in list_of_indices:
    newfile.write(corpus[index])
"""    