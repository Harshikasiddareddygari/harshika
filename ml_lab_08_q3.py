import pandas as pd
import numpy as np

class DecisionTree:
    def __init__(self):
        self.root = None
    
    def fit(self, X, y, impurity='entropy', max_depth=None):
        self.root = self._build_tree(X, y, impurity, max_depth)
    
    def predict(self, X):
        predictions = []
        for index, row in X.iterrows():
            predictions.append(self._predict_row(row))
        return predictions
    
    def _build_tree(self, X, y, impurity, max_depth, depth=0):
        print(f"Building tree at depth {depth}...")
        if depth == max_depth or len(np.unique(y)) == 1:
            print(f"Reached leaf node with depth {depth}")
            return np.argmax(np.bincount(y))
        
        best_feature, best_value = self._find_best_split(X, y, impurity)
        
        if best_feature is None:
            print(f"No best split found at depth {depth}")
            return np.argmax(np.bincount(y))
        
        left_indices = X[best_feature] <= best_value
        right_indices = X[best_feature] > best_value
        
        print(f"Splitting at {best_feature} <= {best_value} (left) and {best_feature} > {best_value} (right)")
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], impurity, max_depth, depth+1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], impurity, max_depth, depth+1)
        
        return {'feature': best_feature, 'value': best_value, 'left': left_subtree, 'right': right_subtree}
    
    def _find_best_split(self, X, y, impurity):
        best_gain = -np.inf
        best_feature = None
        best_value = None
        
        for feature in X.columns:
            if X[feature].dtype in [np.int64, np.float64]:
                values = self._get_numeric_splits(X[feature])
            else:
                values = np.unique(X[feature])
            
            for value in values:
                gain = self._information_gain(X[feature], y, value, impurity)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = value
        
        return best_feature, best_value
    
    def _information_gain(self, feature, y, split_value, impurity):
        left_indices = feature <= split_value
        right_indices = feature > split_value
        
        parent_impurity = self._calculate_impurity(y, impurity)
        left_child_impurity = self._calculate_impurity(y[left_indices], impurity)
        right_child_impurity = self._calculate_impurity(y[right_indices], impurity)
        
        weight_left = len(y[left_indices]) / len(y)
        weight_right = len(y[right_indices]) / len(y)
        
        return parent_impurity - (weight_left * left_child_impurity + weight_right * right_child_impurity)
    
    def _calculate_impurity(self, y, impurity):
        if impurity == 'entropy':
            return self._entropy(y)
        elif impurity == 'gini':
            return self._gini_index(y)
    
    def _entropy(self, y):
        classes = np.unique(y)
        entropy = 0
        for cls in classes:
            p_cls = np.sum(y == cls) / len(y)
            if p_cls != 0:
                entropy -= p_cls * np.log2(p_cls)
        return entropy
    
    def _gini_index(self, y):
        classes = np.unique(y)
        gini = 0
        for cls in classes:
            p_cls = np.sum(y == cls) / len(y)
            gini += p_cls * (1 - p_cls)
        return gini
    
    def _get_numeric_splits(self, feature):
        unique_values = np.unique(feature)
        splits = (unique_values[:-1] + unique_values[1:]) / 2
        return splits
    
    def _predict_row(self, row):
        node = self.root
        while isinstance(node, dict):
            if row[node['feature']] <= node['value']:
                node = node['left']
            else:
                node = node['right']
        return node

# Example usage
if __name__ == "__main__":
    # Load your dataset here
    dataset = pd.read_csv("C:/Users/91630/Downloads/code_only (1).csv")
    
    print("Dataset loaded successfully.")
    print("Columns:", dataset.columns)
    
    # Initialize and fit the Decision Tree
    tree = DecisionTree()
    tree.fit(dataset.drop(columns=['score']), dataset['score'], impurity='entropy', max_depth=5)
    
    # Make predictions
    predictions = tree.predict(dataset.drop(columns=['score']))
    print("Predictions:", predictions)
