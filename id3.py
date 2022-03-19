"""This file implements the id3 algorithm."""
import numpy as np
import vocab
#from collections import Counter
import matplotlib.pyplot as plt


class TreeNode:
    """ This class creates the nodes in our Decision Tree"""
    def __init__(self, node_word="No Word Found", threshold=None, left_node=None, right_node=None, leaf=False, leaf_label=None):
        self.threshold = threshold
        self.node_word = node_word
        self.left_node = left_node
        self.right_node = right_node
        self.leaf = leaf
        self.leaf_label = leaf_label

    def is_terminal(self):
        """ This function helps us identify leaf nodes. It returns true if we are at a leaf node and false otherwise."""
        return self.leaf

    def get_leaf_label(self):
        """ This function returns the label of a leaf node."""
        return self.leaf_label


class DecisionTree:
    """ This class helps us build our Decision Tree"""
    def __init__(self, threshold=0.5, min_samples=2, max_depth=100, number_of_words=None):
        self.threshold = threshold
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.number_of_words = number_of_words
        self.root = None

    def fit(self, train_vec, label_vec):
        """ This function helps us grow our tree"""
        self.number_of_words = train_vec.shape[1]  # the number of words is the same as the words available in our vocab
        self.root = self.build_tree(train_vec, label_vec)  # calling build tree

    def is_valid(self, depth, label_count, sample_count):
        """ This function returns true if the criteria bellow are met, which means we should stop growing our tree."""
        if depth >= self.max_depth or label_count == 1 or sample_count < self.min_samples:
            return True
        return False

    def max_label(self, label_vec):
        """ This function finds the most common label so far."""
        count_label = 0
        for label in label_vec:
            if label == 1:
                count_label += 1
        max_label_c = count_label / len(label_vec) if len(label_vec) != 0 else 0
        return max_label_c

    def find_split(self, column, split_threshold):
        """ Split the data in the given split point."""
        left_samples = np.argwhere(column <= split_threshold).flatten()  # create 1d array with the left samples
        right_samples = np.argwhere(column > split_threshold).flatten()  # create 1d array with the right samples
        return right_samples, left_samples

    def information_gain(self, label_vec, column, split_threshold):
        """ This function calculates the information gain."""
        parent_node_entropy = entropy(label_vec)  # find the parent node's entropy
        left_samples, right_samples = self.find_split(column, split_threshold)  # generate the split
        if len(left_samples) == 0 or len(right_samples) == 0:  # if one node has no samples return 0, there is no information gain
            return 0
        number_of_samples = len(label_vec)
        left_node = len(left_samples)
        right_node = len(right_samples)
        left_node_entropy = entropy(label_vec[left_samples])  # find the left node's entropy
        right_node_entropy = entropy(label_vec[right_samples])  # find the right node's entropy
        prop_left = left_node / number_of_samples  # find the left node's probability
        prop_right = right_node / number_of_samples  # find the right node's probability
        child_node_entropy = (left_node_entropy * prop_left) + (right_node_entropy * prop_right)  # find the child's entropy
        info_gain = parent_node_entropy - child_node_entropy
        return info_gain

    def find_best_word(self, train_vec, label_vec, word_indices):
        """ This function finds the information gain for all the words and returns the index of the word with the greatest information gain."""
        max_info_gain = -1
        best_split_sample = None
        best_split_threshold = None
        for w_index in word_indices:
            column = train_vec[:, w_index]
            thresholds = np.unique(column)
            for thresh in thresholds:
                info_gain = self.information_gain(label_vec, column, thresh)

                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_split_sample = w_index
                    best_split_threshold = thresh

        return best_split_sample, best_split_threshold

    def build_tree(self, train_vec, label_vec, depth=0):
        """ This function builds our tree"""
        sample_count = train_vec.shape[0]  # the number of samples is the number of rows in our vector
        word_count = train_vec.shape[1]  # the number of words is the number of columns in our vector
        label_count = len(np.unique(label_vec))  # the number of labels is the amount of all the different labels
        if self.is_valid(depth, label_count, sample_count):  # check if we should stop building our tree
            leaf_lab = self.max_label(label_vec)  # find the leaf's label (which is the most common label)
            if leaf_lab > self.threshold:  # create a leaf node
                return TreeNode(leaf=True, leaf_label=1)
            else:
                return TreeNode(leaf=True, leaf_label=0)
        else:
            word_indices = np.random.choice(word_count, self.number_of_words, replace=False)  # randomly select the word indices without indices repeating
            best_word, best_threshold = self.find_best_word(train_vec, label_vec, word_indices)  # find what the best word is and what the best point is to split the samples
            left_samples, right_samples = self.find_split(train_vec[:, best_word], best_threshold)  # split the samples
            left_branch = self.build_tree(train_vec[left_samples, :], label_vec[left_samples], depth + 1)  # continue building the tree's left branch
            right_branch = self.build_tree(train_vec[right_samples, :], label_vec[right_samples], depth + 1)  # continue building the tree's right branch
            return TreeNode(best_word, best_threshold, left_branch, right_branch)  # create the current node

    def predict(self, train_vec):
        """ This function helps us traverse our tree."""
        return np.array([self.traverse(sample, self.root) for sample in train_vec])

    def traverse(self, sample, node):
        """ This function traverses our tree."""
        if node.is_terminal():
            return node.get_leaf_label()  # return the leaf's label
        if sample[node.node_word] <= node.threshold:
            return self.traverse(sample, node.left_node)  # continue traversing left
        return self.traverse(sample, node.right_node)   # continue traversing right


def entropy(label_vec):
    """ This function calculates the entropy."""
    _, freqs = np.unique(label_vec, return_counts=True)
    # l=label.shape
    ps = freqs / len(label_vec)
    hc = -np.sum([p * np.log2(p) for p in ps if p > 0])
    return hc


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


def true_negative(real_label, predicted_label):
    """ This function calculates the false negatives."""
    fn = 0
    for j in range(len(real_label)):
        if real_label[j] == predicted_label[j] and real_label[j] == -1:
            fn += 1
    return fn


def precision_metric(real_label, predicted_label):
    """ This function calculates the algorithm's precision."""
    #precision positive
    tp = true_positives(real_label, predicted_label)
    fp = false_positives(real_label, predicted_label)
    if tp != 0 or fp != 0:
        precision_pos = tp / (tp + fp)
    else:
        return 0
    #precision negative
    tn = true_negative(real_label, predicted_label)
    fn = false_negative(real_label, predicted_label)
    if tn != 0 or fn != 0:
        precision_neg = tn / (tn + fn)
    else:
        return 0
    precision = (precision_neg + precision_pos)/2
    return precision


def recall_metric(real_label, predicted_label):
    """ This function calculates the algorithm's recall."""
    # Recall positive
    tp = true_positives(real_label, predicted_label)
    fn = false_negative(real_label, predicted_label)
    if tp != 0 or fn != 0:
        recall_pos = tp / (tp+fn)
    else:
        return 0
    # Recall negative
    tn = true_negative(real_label, predicted_label)
    fp = false_positives(real_label, predicted_label)
    if tn != 0 or fp != 0:
        recall_neg = tn / (tn+fp)
    else:
        return 0
    recall = (recall_neg + recall_pos)/2
    return recall


def f1_metric(real_label, predicted_label):
    """ This function calculates the algorithm's F1 measure."""
    precision = precision_metric(real_label, predicted_label)
    recall = recall_metric(real_label, predicted_label)
    if precision != 0 or recall != 0:
        f1 = 2*precision*recall / (1*precision + recall)
    else:
        return 0
    return f1


negative_list_of_vectors = vocab.convert(vocab.load_file("aclImdb_v1/aclImdb/train/neg"))
positive_list_of_vectors = vocab.convert(vocab.load_file("aclImdb_v1/aclImdb/train/pos"))
X_vector = np.append(negative_list_of_vectors, positive_list_of_vectors, axis=0)

label = vocab.label_vector("aclImdb_v1/aclImdb/train/neg", "aclImdb_v1/aclImdb/train/pos")

vector = np.c_[X_vector, label]  # create numpy array for all the data
np.random.shuffle(vector)  # shuffle the reviews to mix the negative and positive reviews

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
# train_precs = []
dev_precs = []
# train_recalls = []
dev_recalls = []
# train_f1 = []
dev_f1 = []
# create arrays for the predictions, precisions and recalls
for i in range(1, 11):
    i_batch = i*batch
    train = X_vector[:i_batch, :]
    dev = X_vector[i_batch:, :]
    train_label = label[:i_batch]
    dev_label = label[i_batch:]
    id3tree = DecisionTree(max_depth=10)
    id3tree.fit(train, train_label)
    train_predictions = id3tree.predict(train)
    dev_predictions = id3tree.predict(dev)
    train_acc = accuracy_metric(train_label, train_predictions)
    dev_acc = accuracy_metric(dev_label, dev_predictions)
    train_accs.append(train_acc)
    dev_accs.append(dev_acc)

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
    id3tree = DecisionTree(threshold=threshold, max_depth=10)
    id3tree.fit(train, train_label)
    dev_predictions = id3tree.predict(dev)
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
plt.plot(list(range(len(dev_precs))), dev_precs, '^', label="PRECISION")
plt.plot(list(range(len(dev_recalls))), dev_recalls, label="RECALL")
leg = plt.legend(loc='lower center', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show()

ax = plt.subplot(111)
t1 = np.arange(0.0, 1.0, 0.01)
plt.plot(list(range(len(dev_precs))), dev_precs, '^', label="PRECISION")
plt.plot(list(range(len(dev_recalls))), dev_recalls, label="RECALL")
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
random_forest = DecisionTree(max_depth=13)
random_forest.fit(train, train_label)
test_predictions = random_forest.predict(test)
print(accuracy_metric(test_label, test_predictions)) 
"""
