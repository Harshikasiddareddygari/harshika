from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
class1_data = pd.read_csv('C:/Users/91630/Downloads/code_comm.csv')
class2_data = pd.read_csv('C:/Users/91630/Downloads/code_only.csv')

# Assuming that the target variable is in a column named 'target', adjust accordingly
class1_data['target'] = 0
class2_data['target'] = 1

# Concatenate the two datasets
combined_data = pd.concat([class1_data, class2_data], ignore_index=True)

# Impute missing values with mean (you can choose a different strategy)
imputer = SimpleImputer(strategy='mean')
combined_data_imputed = pd.DataFrame(imputer.fit_transform(combined_data), columns=combined_data.columns)

# Split data into features (X) and target variable (y)
X = combined_data_imputed.drop('target', axis=1)
y = combined_data_imputed['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNeighborsClassifier and fit the model
knn = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors
knn.fit(X_train, y_train)

# Make predictions on the test set
predicted_classes = knn.predict(X_test)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Precision, recall, and F1-score
precision = precision_score(y_test, predicted_classes, average='weighted')
recall = recall_score(y_test, predicted_classes, average='weighted')
f1 = f1_score(y_test, predicted_classes, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Display the shapes of training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Training accuracy
training_accuracy = knn.score(X_train, y_train)
print("Training Accuracy:", training_accuracy)

# Testing accuracy
testing_accuracy = knn.score(X_test, y_test)
print("Testing Accuracy:", testing_accuracy)

