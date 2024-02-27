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
# Different values of k
k_values = [1, 3, 5, 7]

# Create subplots for each value of k
plt.figure(figsize=(15, 10))
for i, k in enumerate(k_values, 1):
    # Perform kNN classification
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)

    # Create subplot
    plt.subplot(2, 2, i)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Paired, marker='.', s=10)
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='blue', marker='o', label='Class 0 (Training)')
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='orange', marker='o', label='Class 1 (Training)')

    plt.title(f'kNN Classification (k={k})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

# Adjust layout
plt.tight_layout()
plt.show()

#k=1: The decision boundaries will be highly influenced by individual data points. The model may capture noise in the data and might lead to overfitting.

#k=3: The decision boundaries will be smoother compared to k=1, as the classification takes into account the labels of three nearest neighbors. This can provide a more generalized result.

#k=5: Further smoothing of the decision boundaries. The influence of individual data points is reduced, leading to a more generalized classification.

#k=7: The decision boundaries become even smoother. The model considers a larger number of neighbors, resulting in a more robust and less sensitive classification.
