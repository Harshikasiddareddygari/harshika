import numpy as np
import pandas as pd
from collections import Counter

class DecisionTree:
    def __init__(self):
        self.tree = None

    def calculate_entropy(self, data):
        labels = data[:, -1]
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        probabilities = label_counts / len(labels)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def calculate_information_gain(self, data, feature_index):
        entropy_parent = self.calculate_entropy(data)
        unique_values = np.unique(data[:, feature_index])
        weighted_entropy_children = 0

        for value in unique_values:
            subset = data[data[:, feature_index] == value]
            probability = len(subset) / len(data)
            weighted_entropy_children += probability * self.calculate_entropy(subset)

        information_gain = entropy_parent - weighted_entropy_children
        return information gain

    def find_best_split(self, data):
        num_features = data.shape[1] - 1
        best_feature = None
        best_information_gain = -1

        for i in range(num_features):
            information_gain = self.calculate_information_gain(data, i)
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feature = i

        return best_feature

    def fit(self, data):
        def build_tree(data):
            if len(np.unique(data[:, -1])) == 1:
                return Counter(data[:, -1]).most_common(1)[0][0]

            best_feature = self.find_best_split(data)
            tree = {best_feature: {}}
            unique_values = np.unique(data[:, best_feature])

            for value in unique_values:
                subset = data[data[:, best_feature] == value]
                tree[best_feature][value] = build_tree(subset)

            return tree

        self.tree = build_tree(data)

    def predict(self, data):
        def classify(instance, tree):
            if not isinstance(tree, dict):
                return tree

            feature_index = list(tree.keys())[0]
            subtree = tree[feature_index]
            value = instance[feature_index]
            if value not in subtree:
                return None

            return classify(instance, subtree[value])

        predictions = [classify(instance, self.tree) for instance in data]
        return predictions

#dataset (CSV file)
file_path = 'C:/Users/91630/Downloads/code_only.csv'
data = pd.read_csv(file_path).values  # Assuming the last column is the score

dt = DecisionTree()
dt.fit(data)

print("Root node feature:", dt.find_best_split(data))
