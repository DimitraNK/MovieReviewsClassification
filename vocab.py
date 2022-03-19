"""This file creates the vocabulary we need using imdb.vocab, our dataset and the following functions"""
import glob
import os
from collections import Counter
from csv import reader
import numpy as np
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
import string


def load_review(filename):
    """This function loads a review, using functions open, read and close, and takes it's content and assigns it to the variable text."""
    file = open(filename, 'r', encoding="UTF-8")
    text = file.read()
    file.close()
    return text


def cleaner(review):
    """This function cleans a review and returns the valuable words. """
    words = review.split()  # remove white space
    words = [word.lower() for word in words]  # turn every word to lower case to make sure each review's words will match a word in our vocab
    words = [word for word in words if word.isalpha()]  # only keep alphabet letters using isalpha() function
    punct = set(string.punctuation)  # create set for punctuation
    stop_words = set(stopwords.words('english'))  # find common words in english that aren't necessary for our vocab
    l = lambda w: w.replace("won't", "will not")
    words = list(map(l, words))
    l = lambda w: w.replace("n't", " not")
    words = list(map(l, words))
    l = lambda w: w.replace("'ll", " will")
    words = list(map(l, words))
    l = lambda w: w.replace("'re", " are")
    words = list(map(l, words))
    l = lambda w: w.replace("'ve", " have")
    words = list(map(l, words))
    l = lambda w: w.replace("'m", " am")
    words = list(map(l, words))
    l = lambda w: w.replace("'s", "")
    words = list(map(l, words))
    stop_list = {"i have", "you have", "we have", "they have", "i will", "you will", "he will", "she will", "it will", "we will", "they will", "they are", "you are", "we are", "i am", "will not", "have not", "am not", "are not", "has not"}
    stop_words.update(stop_list)   # add words that are common and stopwords might not be able to remove from our reviews
    words = [w for w in words if w not in stop_words and w not in punct]  # remove punctuation and words in stopwords
    words = [word for word in words if len(word) > 1]  # remove words that are longer than one letter
    return words   # return the rest of the words


def add_to_vocab(filename, vocab):
    """ This function adds words from the reviews to our vocab by calling the cleaner function (which gets rid of words we don't need in our vocab)."""
    review = load_review(filename)  # load review to clean
    words = cleaner(review)
    vocab.update(words)  # update words by adding new words and increasing words' frequencies


def process_review(directory, vocab):
    """ This function adds the words of every review (.txt file) in a directory to our vocab."""
    for filename in listdir(directory):  # for every file in the list of files
        if not filename.endswith(".txt"):  # if it's not a .txt file
            continue
        path = directory + '/' + filename  # create path
        add_to_vocab(path, vocab)  # call add_to_vocab


def save_list(lines, filename):
    """ This function saves our vocab to a .txt file."""
    data = '\n'.join(lines)
    file = open(filename, 'w', encoding="UTF-8")
    file.write(data)
    file.close()


def load_file(path):
    """ This function creates a list of all the words in the reviews."""
    list_of_reviews = list()
    for filename in glob.glob(os.path.join(path, '*.txt')):
        with open(os.path.join(os.getcwd(), filename), 'r', encoding="UTF-8") as file:
            file_reader = reader(file)
            review = ""
            words = list()
            for row in file_reader:
                if not row:
                    continue
                for i in row:
                    review += i.lower()  # every word is lower case to match the vocab
            words.append(review.split())
        list_of_reviews.append(words)
    return list_of_reviews


def vocabulary_vector(list_of_reviews):
    """ This function creates a vector. In the vector for every review, for every word in the vocab, we use 0 when the word is not in the review and 1 if it is."""
    list_of_vectors = list()
    text_file = open('vocab.txt', 'r', encoding="UTF-8")
    voc_word = text_file.read().splitlines()  # assign to voc_word every word we have in our vocab
    for row in list_of_reviews:
        for j in row:
            vector = list()
            for v in voc_word:
                done = 0
                for word in j:
                    if word == v:
                        vector.append(1)  # this word is in the review
                        done = 1
                        break  # if we find the word is in the review at least once we don't need to look for it again
                if done == 0:
                    vector.append(0)  # this word is not in the review
            list_of_vectors.append(vector)
    return list_of_vectors

def convert(list_of_reviews):
    """ This function converts our vector to a numpy array."""
    list_rev = vocabulary_vector(list_of_reviews)
    vector_arr = np.array([np.array(x) for x in list_rev])
    return vector_arr

def label_vector(path_pos, path_neg):
    """ This function creates a label vector,which labels positive reviews with 1 and negative reviews with -1."""
    neg = [f for f in listdir(path_neg) if isfile(join(path_neg, f))]  # create a list with all the negative reviews
    pos = [f for f in listdir(path_pos) if isfile(join(path_pos, f))]  # create a list with all the positive reviews
    temp = []
    for i in range(len(neg)):  # for every negative review add -1 to temp
        temp.append(-1)
    for i in range(len(pos)):  # for every positive review add 1 to temp
        temp.append(1)
    label = np.array(temp)
    return label

# define vocab
vocab = Counter()
# add all docs to vocab
process_review("aclImdb_v1/aclImdb/train/neg", vocab)
process_review("aclImdb_v1/aclImdb/train/pos", vocab)

occmax = 12010  # add words that occur in the reviews at most 12010 times
occmin = 500    # add words that occur in the reviews at least 350 times

voc_words = [k for k, c in vocab.items() if (occmax > c > occmin)]  # create the list of words that we will have in our vocab
save_list(voc_words, 'vocab.txt')  # save every word to a .txt


"""
negative_list_of_vectors = convert(load_file("aclImdb_v1/aclImdb/train/neg"))
positive_list_of_vectors = convert(load_file("aclImdb_v1/aclImdb/train/pos"))
label_vector("aclImdb_v1/aclImdb/train/neg", "aclImdb_v1/aclImdb/train/pos")"""
