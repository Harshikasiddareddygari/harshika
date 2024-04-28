import numpy as np
import pandas as pd

class DecisionTreeRootNodeDetector:
    def __init__(self):
        pass

    def entropy(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return -np.sum(probabilities * np.log2(probabilities))

    def information_gain(self, parent_labels, splits_labels):
        parent_entropy = self.entropy(parent_labels)
        splits_entropy = 0
        total_samples = sum(len(split) for split in splits_labels)
        for split_labels in splits_labels:
            split_weight = len(split_labels) / total_samples
            splits_entropy += split_weight * self.entropy(split_labels)
        return parent_entropy - splits_entropy

    def find_root_node(self, features, labels):
        best_info_gain = -1
        best_feature_index = None

        for feature_index in range(features.shape[1]):
            feature_values = features[:, feature_index]
            unique_values = np.unique(feature_values)
            splits_labels = []
            for value in unique_values:
                split_indices = np.where(feature_values == value)[0]
                split_labels = labels[split_indices]
                splits_labels.append(split_labels)

            info_gain = self.information_gain(labels, splits_labels)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_index = feature_index

        return best_feature_index

# Load dataset
dataset = pd.read_csv("C:/Users/91630/Downloads/code_only (1).csv")  # Replace 'your_dataset.csv' with the path to your CSV file

# Convert non-numeric columns to strings
for col in dataset.columns:
    if not pd.api.types.is_numeric_dtype(dataset[col]):
        dataset[col] = dataset[col].astype(str)

# Extract features and labels
features = dataset.drop(columns=['indicator']).values
labels = dataset['indicator'].values

# Initialize and use the DecisionTreeRootNodeDetector
detector = DecisionTreeRootNodeDetector()
root_node_index = detector.find_root_node(features, labels)
print("Root node feature index:", root_node_index)
