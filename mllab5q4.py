import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate training data
np.random.seed(42)
X_train = np.random.rand(100, 2) * 10
y_train = np.random.choice([0, 1], size=100)

# Generate test data
x_range = np.arange(0, 10.1, 0.1)
y_range = np.arange(0, 10.1, 0.1)
X_test = np.array(np.meshgrid(x_range, y_range)).T.reshape(-1, 2)

# Perform kNN classification
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

# Create scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Paired, marker='.', s=10)
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='blue', marker='o', label='Class 0 (Training)')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='orange', marker='o', label='Class 1 (Training)')

plt.title('kNN Classification of Test Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
