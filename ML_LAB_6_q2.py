import numpy as np
import matplotlib.pyplot as plt

# Define the AND gate input-output pairs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])

# Initial weights
W = np.array([10, 0.2, -0.75])

# Learning rate
alpha = 0.05

# Bi-Polar Step function
def bipolar_step_function(x):
    return 1 if x >= 0 else -1

# Sigmoid function
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

# ReLU function
def relu_function(x):
    return max(0, x)

# Modify perceptron function to use bipolar step function
def perceptron_bipolar(x, w):
    return bipolar_step_function(np.dot(x, w))

# Modify perceptron function to use sigmoid function
def perceptron_sigmoid(x, w):
    return sigmoid_function(np.dot(x, w))

# Modify perceptron function to use ReLU function
def perceptron_relu(x, w):
    return relu_function(np.dot(x, w))

# Calculate sum-square error
def calculate_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

# Training function
def train_perceptron(perceptron_func, W):
    error_values = []
    epochs = 0
    max_epochs = 1000
    convergence_error = 0.002

    while True:
        error = 0
        for i in range(len(X)):
            y_pred = perceptron_func(np.insert(X[i], 0, 1), W)
            delta = Y[i] - y_pred
            W += alpha * delta * np.insert(X[i], 0, 1)
            error += calculate_error(Y[i], y_pred)
        error_values.append(error)
        epochs += 1
        if error <= convergence_error or epochs >= max_epochs:
            break

    return epochs, error_values

# Train with Bi-Polar Step function
epochs_bipolar, error_values_bipolar = train_perceptron(perceptron_bipolar, W.copy())

# Train with Sigmoid function
epochs_sigmoid, error_values_sigmoid = train_perceptron(perceptron_sigmoid, W.copy())

# Train with ReLU function
epochs_relu, error_values_relu = train_perceptron(perceptron_relu, W.copy())

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs_bipolar + 1), error_values_bipolar, label='Bi-Polar Step')
plt.plot(range(1, epochs_sigmoid + 1), error_values_sigmoid, label='Sigmoid')
plt.plot(range(1, epochs_relu + 1), error_values_relu, label='ReLU')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs Error for Different Activation Functions')
plt.legend()
plt.grid(True)
plt.show()

print("Number of epochs needed for convergence (Bi-Polar Step):", epochs_bipolar)
print("Number of epochs needed for convergence (Sigmoid):", epochs_sigmoid)
print("Number of epochs needed for convergence (ReLU):", epochs_relu)
