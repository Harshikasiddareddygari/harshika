import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load dataset
dataset = pd.read_csv("C:/Users/91630/Downloads/code_comm (1).csv")  # Adjust the path to your dataset

# Convert non-numeric columns to strings
for col in dataset.columns:
    if not pd.api.types.is_numeric_dtype(dataset[col]):
        dataset[col] = dataset[col].astype(str)

# Map 'code_only' to 0 and 'code_comm' to 1
dataset['indicator'] = dataset['indicator'].map({'code_only': 0, 'code_comm': 1})

# Preprocess the data
X = dataset.drop(columns=['indicator'])
y = dataset['indicator']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define perceptron model
perceptron = Perceptron()

# Define hyperparameters for perceptron
param_dist_perceptron = {'alpha': uniform(0.0001, 0.1)}

# Initialize RandomizedSearchCV for perceptron
perceptron_search = RandomizedSearchCV(perceptron, param_distributions=param_dist_perceptron, n_iter=10, cv=5, random_state=42)

# Fit RandomizedSearchCV for perceptron
perceptron_search.fit(X_train_scaled, y_train)

# Get best perceptron model
best_perceptron = perceptron_search.best_estimator_

# Predict using best perceptron model
y_pred_perceptron = best_perceptron.predict(X_test_scaled)

# Calculate accuracy of perceptron model
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
print("Perceptron Accuracy:", accuracy_perceptron)
print("Best Perceptron Hyperparameters:", perceptron_search.best_params_)


# Define MLPClassifier model
mlp_classifier = MLPClassifier(max_iter=500, random_state=42)

# Define hyperparameters for MLPClassifier
param_dist_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': uniform(0.0001, 0.1)
}

# Initialize RandomizedSearchCV for MLPClassifier
mlp_search = RandomizedSearchCV(mlp_classifier, param_distributions=param_dist_mlp, n_iter=10, cv=5, random_state=42)

# Fit RandomizedSearchCV for MLPClassifier
mlp_search.fit(X_train_scaled, y_train)

# Get best MLPClassifier model
best_mlp = mlp_search.best_estimator_

# Predict using best MLPClassifier model
y_pred_mlp = best_mlp.predict(X_test_scaled)

# Calculate accuracy of MLPClassifier model
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print("MLPClassifier Accuracy:", accuracy_mlp)
print("Best MLPClassifier Hyperparameters:", mlp_search.best_params_)

# Define classifiers
classifiers = {
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'XGBoost': XGBClassifier(),
    'Naive Bayes': GaussianNB(),
    'CatBoost': CatBoostClassifier()
}

# Initialize lists to store results
accuracy_list = []
precision_list = []
recall_list = []
f1_score_list = []

# Iterate over classifiers
for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_score_list.append(f1)

# Create DataFrame
results_df = pd.DataFrame({
    'Classifier': list(classifiers.keys()),
    'Accuracy': accuracy_list,
    'Precision': precision_list,
    'Recall': recall_list,
    'F1-score': f1_score_list
})

# Display results
print(results_df)
