from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# AND Gate logic
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# XOR Gate logic
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Normalize input data
scaler = StandardScaler()
X_and_scaled = scaler.fit_transform(X_and)
X_xor_scaled = scaler.fit_transform(X_xor)

# Create MLPClassifier for AND Gate
mlp_and = MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', max_iter=2000, learning_rate='adaptive', random_state=42)

# Train the MLPClassifier for AND Gate
mlp_and.fit(X_and_scaled, y_and)

# Create MLPClassifier for XOR Gate
mlp_xor = MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', max_iter=2000, learning_rate='adaptive', random_state=42)

# Train the MLPClassifier for XOR Gate
mlp_xor.fit(X_xor_scaled, y_xor)

# Test the models
test_data_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_data_and_scaled = scaler.transform(test_data_and)
predictions_and = mlp_and.predict(test_data_and_scaled)

test_data_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_data_xor_scaled = scaler.transform(test_data_xor)
predictions_xor = mlp_xor.predict(test_data_xor_scaled)

# Print the results
print("MLPClassifier Predictions for AND Gate:")
print(predictions_and)

print("\nMLPClassifier Predictions for XOR Gate:")
print(predictions_xor)
