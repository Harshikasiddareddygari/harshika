import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your dataset
file_path = 'C:\\Users\\lehas\\OneDrive\\Documents\\B-TECH_AIE_D\\sem_04\\ML\\DATA\\code_only.csv'
project_data = pd.read_csv(file_path)

# Assuming your target variable is named 'target_column', replace it with the actual column name in your dataset
X = project_data.drop('target_column', axis=1)
y = project_data['target_column']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize input data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create MLPClassifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Train the MLPClassifier
mlp_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions = mlp_classifier.predict(X_test_scaled)

# Evaluate the performance
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_report_str = classification_report(y_test, predictions)

# Print the results
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report_str)
