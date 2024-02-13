import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_excel("C:\\Users\\91630\\Downloads\\Lab Session1 Data.xlsx", sheet_name="Purchase data")

# Mark customers with payments above Rs. 200 as RICH and others as POOR
data['Class'] = np.where(data['Payment (Rs)'] > 200, 'RICH', 'POOR')

# Convert 'Class' column to numerical values (1 for RICH, 0 for POOR)
data['Class'] = data['Class'].map({'RICH': 1, 'POOR': 0})

# Segregate data into matrices A & C
A = data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = data['Class'].values

# Printing matrices A & C
print("Matrix A:\n", A)
print("Matrix C:\n", C)

# Dimensionality of the vector space
dimensionality = A.shape[1]
print("Dimensionality of the vector space:", dimensionality)

# Number of vectors in the vector space
num_vectors = A.shape[0]
print("Number of vectors in the vector space:", num_vectors)

# Rank of Matrix A
rank_A = np.linalg.matrix_rank(A)
print("Rank of Matrix A:", rank_A)

# Pseudo-inverse of A
A_pseudo_inv = np.linalg.pinv(A)

# Cost of each product available for sale
cost_per_product = np.dot(A_pseudo_inv, C)
print("Cost of each product available for sale:", cost_per_product)

# Model vector X for predicting product costs
model_vector_X = np.dot(A_pseudo_inv, C)
print("Model vector X for predicting product costs:", model_vector_X)


# Splitting data into features (A) and target variable (C)
X_train, X_test, y_train, y_test = train_test_split(A, C, test_size=0.2, random_state=42)

# Initializing and training the classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting on the test set
y_pred = classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the classifier model:", accuracy)
