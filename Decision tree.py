import numpy as np
import string
import json
import sys
import pickle
import pandas as pd
import random

#################### DECISION TREE MODEL #####################


class Dtree(object):
    def __init__(self, generate_new_word_token=False, word_freq_threshold=150):
        self.data_preprecessing = Data_preprocessing(
            generate_new_word_token, word_freq_threshold)

    def information_gain(self, contingency_matrix):
        '''
        contingency_matrix
         - shape: |feature_group| x |classifications|
         - Multivariate frequency distribution of the variables
        '''
        def get_entropies(contingency_matrix):
            p = 1e-200 + contingency_matrix / \
                contingency_matrix.sum(axis=1, keepdims=True)
            return (- p * np.log2(p)).sum(axis=1).flatten()

        # Parent entropy
        parent_entropy = get_entropies(
            contingency_matrix.sum(axis=0, keepdims=True))

        # Weighted child entory
        child_entropies = get_entropies(contingency_matrix)
        weights = (contingency_matrix.sum(axis=1) /
                   contingency_matrix.sum()).flatten()
        weighted_entropies_child = weights @ child_entropies

        # Information gain - IG
        IG = parent_entropy - weighted_entropies_child
        return IG

    def get_contingency_matrix(self, features, labels):
        '''
        features = binary vector
        labels =  multi-nary vector
        '''
        return pd.crosstab(features, labels).values

    def fit(self, X, y):
        X_processed = self.data_preprecessing(X)
        features = self.data_preprecessing.words.copy()
        examples = (X_processed, y)
        parent_examples = None
        self.dtree = self.DTL(examples, features, parent_examples)

    def DTL(self, examples, features, parent_examples,
            max_tree_depth=100, min_node_size=10,
            max_features='sqrt',
            random_state=551
            ):
        '''
        The DTL is modified from DECISION-TREE-LEARNING algorithm on the course textbook
        examples: (X, y)
        features: words
        return: a tree
        '''
        def plurality_value(examples):
            X, y = examples
            freqs = [((y == ex).sum(), ex) for ex in set(y)]
            highest_freq = max(freqs)[0]
            highest_freq_index = np.where(
                np.array([freq[0] for freq in freqs]) == highest_freq)
            if len(highest_freq_index) > 1:
                return freqs[random.choice(highest_freq_index)][1]
            else:
                return max(freqs)[1]

        def importance(examples, feature):
            X, y = examples
            index = features.index(feature)
            contingency_matrix = self.get_contingency_matrix(
                X[:, index], y)
            information_gain = self.information_gain(contingency_matrix)
            return information_gain

        X, y = examples
        if X.shape[0] == 0 or len(X[0]) == 0:
            return Node(plurality_value(parent_examples))
        elif X.shape[0] < min_node_size:
            return Node(plurality_value(examples))
        elif len(set(y)) == 1:
            return Node(y[0])
        elif len(features) == 0:
            return Node(plurality_value(examples))
        elif max_tree_depth == 0:
            return Node(plurality_value(examples))
        else:
            n = int(np.sqrt(len(features)))
            np.random.seed(random_state)
            best_feature = max([
                (importance(examples, feature), feature)
                for feature in features
                # for feature in np.random.choice(features, n)
            ])[1]
            tree = Node(feature=best_feature)
            best_feature_index = features.index(best_feature)
            features.remove(best_feature)
            new_X = np.delete(X, best_feature_index, axis=1)
            max_tree_depth -= 1
            for v in [0, 1]:  # sorted(np.unique(X[:, best_feature_index])):
                exs_index = X[:, best_feature_index] == v
                exs = (new_X[exs_index, :], y[exs_index])
                subtree = self.DTL(exs, features, examples, max_tree_depth)
                tree.child_node[v] = subtree
                tree.child_node[v].parent = tree
            return tree

    def predict(self, test_X):
        predicted_loc = []
        test_X_processed = self.data_preprecessing(test_X)
        for tweet in test_X_processed:
            node = self.dtree
            while not node.is_leaf:
                feature = node.feature
                feature_index = self.data_preprecessing.words.copy().index(feature)
                if tweet[feature_index]:
                    node = node.child_node[1]
                else:
                    node = node.child_node[0]

            predicted_loc.append(node.feature)

        return predicted_loc

    def model_info(self, level_thres=10):
        nodes = [self.dtree]
        recorder = []
        while nodes:
            node = nodes.pop()
            if node.tree_level < level_thres:
                parent = node.parent.feature if not node.is_root else None
                recorder.append(
                    (f'Level: {node.tree_level}', f'Node: {node.feature}', f'Parent: {parent}', f'Is_leaf: {node.is_leaf}'))
                nodes.extend([node.child_node[key]
                              for key in node.child_node])
        return '\n'.join([str(recd).replace("\'", "").replace("(", "").replace(")", "") for recd in recorder])


class Node(object):
    def __init__(self, feature):
        self.feature = feature
        self.value = None
        self.child_node = {}
        self.parent = None

    @ property
    def is_root(self):
        return self.parent == None

    @ property
    def is_leaf(self):
        return self.child_node == {}

    @ property
    def tree_level(self):
        count = 0
        while self.parent != None:
            count += 1
            self = self.parent
        return count
