"""This file implements the Random Forest algorithm."""
import numpy as np
import vocab
from collections import Counter
from id3 import DecisionTree
import matplotlib.pyplot as plt


class RandomForest:
    """ This class helps us create a random forest."""
    def __init__(self, number_of_trees=10, min_samples=2, max_depth=100, number_of_words=None, threshold=0.5):
        self.number_of_trees = number_of_trees
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.number_of_words = number_of_words
        self.threshold = threshold
        self.tree_array = []

    def fit(self, X_vector, label_vec):
        """ This function helps us grow our trees"""
        self.tree_array = []
        for _ in range(self.number_of_trees):
            tree = DecisionTree(min_samples=self.min_samples, max_depth=self.max_depth, number_of_words=self.number_of_words)  # create each tree using id3's Decision tree class
            X_vector_sample, label_vec_sample = create_sample(X_vector, label_vec)
            tree.fit(X_vector_sample, label_vec_sample)
            self.tree_array.append(tree)  # add the current tree to the tree list

    def predict(self, X_vector):
        """ This function finds a prediction for each of our trees."""
        tree_predictions = np.array([tree.predict(X_vector) for tree in self.tree_array])
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        label_predictions = [max_label(tree_predictions) for tree_predictions in tree_predictions]
        return np.array(label_predictions)


def create_sample(X_vector, label_vec):
    """ This function creates a sample out of all the word indices."""
    number_of_samples = X_vector.shape[0]
    word_indices = np.random.choice(number_of_samples, size=number_of_samples, replace=True)
    return X_vector[word_indices], label_vec[word_indices]


def max_label(label_vec):
    """ This function finds the most common label."""
    count_label = Counter(label_vec)
    max_label_c = count_label.most_common(1)[0][0]
    return max_label_c


def accuracy_metric(real_label, predicted_label):
    """ This function calculates the algorithm's accuracy."""
    if len(real_label) != 0:
        accuracy = np.sum(real_label == predicted_label) / len(real_label)
    else:
        return 0
    return accuracy


def true_positives(real_label, predicted_label):
    """ This function calculates the true positives."""
    tp = 0
    for j in range(len(real_label)):
        if real_label[j] == predicted_label[j] and real_label[j] == 1:
            tp += 1
    return tp


def false_positives(real_label, predicted_label):
    """ This function calculates the false positives."""
    fp = 0
    for j in range(len(real_label)):
        if real_label[j] != predicted_label[j] and real_label[j] == -1:
            fp += 1
    return fp


def false_negative(real_label, predicted_label):
    """ This function calculates the false negatives."""
    fn = 0
    for j in range(len(real_label)):
        if real_label[j] != predicted_label[j] and real_label[j] == 1:
            fn += 1
    return fn


def precision_metric(real_label, predicted_label):
    """ This function calculates the algorithm's precision."""
    tp = true_positives(real_label, predicted_label)
    fp = false_positives(real_label, predicted_label)
    if tp != 0 or fp !=0:
        precision = tp / (tp+fp)
    else:
        return 0
    return precision


def recall_metric(real_label, predicted_label):
    """ This function calculates the algorithm's recall."""
    tp = true_positives(real_label, predicted_label)
    fn = false_negative(real_label, predicted_label)
    if tp != 0 or fn != 0:
        recall = tp / (tp+fn)
    else:
        return 0
    return recall


def f1_metric(real_label, predicted_label):
    """ This function calculates the algorithm's F1 measure."""
    precision = precision_metric(real_label, predicted_label)
    recall = recall_metric(real_label, predicted_label)
    if precision != 0 or recall != 0:
        f1 = 2*precision*recall / 1*precision + recall
    else:
        return 0
    return f1


negative_list_of_vectors = vocab.convert(vocab.load_file("aclImdb_v1/aclImdb/train/neg"))
positive_list_of_vectors = vocab.convert(vocab.load_file("aclImdb_v1/aclImdb/train/pos"))
X_vector = np.append(negative_list_of_vectors, positive_list_of_vectors, axis=0)

label = vocab.label_vector("aclImdb_v1/aclImdb/train/neg", "aclImdb_v1/aclImdb/train/pos")

vector = np.c_[X_vector, label]  # create numpy array for all the reviews
np.random.shuffle(vector)  # shuffle the eviews to mix positive and negative reviews

# read vocab
V = open('vocab.txt', 'r', errors='ignore')
Vocab = []
for line in V:
    Vocab.append(line.strip())
label = vector[:, len(Vocab)]  # create the vector for the labels
X_vector = np.delete(vector, len(Vocab), 1)

batch = 2500  # one 10th of our dataset
train_accs = []
dev_accs = []
dev_precs = []
dev_recalls = []
dev_f1 = []
# create arrays for the predictions, precisions and recalls
for i in range(1, 11):
    i_batch = i*batch
    train = X_vector[:i_batch, :]
    dev = X_vector[i_batch:, :]
    train_label = label[:i_batch]
    dev_label = label[i_batch:]
    random_forest = RandomForest(number_of_trees=3, max_depth=11)
    random_forest.fit(train, train_label)
    train_predictions = random_forest.predict(train)
    dev_predictions = random_forest.predict(dev)
    train_acc = accuracy_metric(train_label, train_predictions)
    dev_acc = accuracy_metric(dev_label, dev_predictions)
    train_accs.append(dev_acc)
    dev_accs.append(train_acc)

max = 0
i_max = -1
for i in range(0, 10):
    if max < dev_accs[i]:
        max = dev_accs[i]
        i_max = i
# highest accuracy
i_batch = i_max * batch
train = X_vector[:i_batch, :]
dev = X_vector[i_batch:, :]
train_label = label[:i_batch]
dev_label = label[i_batch:]
for i in range(1, 10):
    threshold = i/10
    random_forest = RandomForest(number_of_trees=3, max_depth=11, threshold=threshold)
    random_forest.fit(train, train_label)
    dev_predictions = random_forest.predict(dev)
    dev_prec = precision_metric(dev_label, dev_predictions)
    dev_precs.append(dev_prec)
    dev_recall = recall_metric(dev_label, dev_predictions)
    dev_recalls.append(dev_recall)
    devf1 = f1_metric(dev_label, dev_predictions)
    dev_f1.append(devf1)

print("ACCURACY VALUES")
print("TRAIN: ", train_accs)
print("DEVELOPMENT: ", dev_accs)
print("PRECISION VALUES")
print("PRECISION: ", dev_precs)
print("RECALL VALUES")
print("RECALL: ", dev_recalls)
print("F-1 VALUES")
print("F-1: ", dev_f1)

# ACCURACY
plt.rcParams['figure.figsize'] = [15, 10]

ax = plt.subplot(111)
t1 = np.arange(0.0, 1.0, 0.01)
plt.plot(list(range(len(train_accs))), train_accs, '^', label="TRAIN")
plt.plot(list(range(len(dev_accs))), dev_accs, label="DEV")
leg = plt.legend(loc='lower center', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show()

ax = plt.subplot(111)
t1 = np.arange(0.0, 1.0, 0.01)
plt.plot(list(range(len(train_accs))), train_accs, '^', label="TRAIN")
plt.plot(list(range(len(dev_accs))), dev_accs, label="DEV")
leg = plt.legend(loc='lower center', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.ylim(0.0, 1.1)
plt.show()

# PRECISION - RECALL
plt.rcParams['figure.figsize'] = [15, 10]

ax = plt.subplot(111)
t1 = np.arange(0.0, 1.0, 0.01)
plt.plot(list(range(len(dev_precs))), dev_precs, '^', label="TRAIN")
plt.plot(list(range(len(dev_recalls))), dev_recalls, label="DEV")
leg = plt.legend(loc='lower center', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show()

ax = plt.subplot(111)
t1 = np.arange(0.0, 1.0, 0.01)
plt.plot(list(range(len(dev_precs))), dev_precs, '^', label="TRAIN")
plt.plot(list(range(len(dev_recalls))), dev_recalls, label="DEV")
leg = plt.legend(loc='lower center', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.ylim(0.0, 1.1)
plt.show()

""" Find Accuracy for test data:
negative_list_of_vectors = vocab.convert(vocab.load_file("aclImdb_v1/aclImdb/train/neg"))
positive_list_of_vectors = vocab.convert(vocab.load_file("aclImdb_v1/aclImdb/train/pos"))
X_vector_train = np.append(negative_list_of_vectors, positive_list_of_vectors, axis=0)

label_train = vocab.label_vector("aclImdb_v1/aclImdb/train/neg", "aclImdb_v1/aclImdb/train/pos")

negative_list_of_test = vocab.convert(vocab.load_file("aclImdb_v1/aclImdb/test/neg"))
positive_list_of_test = vocab.convert(vocab.load_file("aclImdb_v1/aclImdb/test/pos"))
X_vector_test = np.append(negative_list_of_test, positive_list_of_test, axis=0)

label_test = vocab.label_vector("aclImdb_v1/aclImdb/test/neg", "aclImdb_v1/aclImdb/test/pos")
label = np.concatenate([label_train, label_test])
X_vetcor = np.concatenate((X_vector_train, X_vector_test), axis=0)

vector = np.c_[X_vector, label]  # create numpy array for all the reviews
np.random.shuffle(vector)  # shuffle the eviews to mix positive and negative reviews

# read vocab
V = open('vocab.txt', 'r', errors='ignore')
Vocab = []
for line in V:
    Vocab.append(line.strip())
label = vector[:, len(Vocab)]  # create the vector for the labels
X_vector = np.delete(vector, len(Vocab), 1)
train = X_vector[:25000, :]
test = X_vector[25000:, :]
train_label = label[:25000]
test_label = label[25000:]
random_forest = RandomForest(number_of_trees=3, max_depth=11)
random_forest.fit(train, train_label)
test_predictions = random_forest.predict(test)
print(accuracy_metric(test_label, test_predictions)) 
"""
