import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return 1 if x >= 0 else -1

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
    return max(0, x)

def train_perceptron(X, y, activation_function, learning_rate, max_epochs=1000, convergence_error=0.002):
    # Initialize weights
    W0 = 10
    W1 = 0.2
    W2 = -0.75

    error_list = []
    epochs = 0

    for epoch in range(max_epochs):
        total_error = 0

        for i in range(len(X)):
            # Forward pass
            net_input = W0 + W1 * X[i, 0] + W2 * X[i, 1]
            predicted_output = activation_function(net_input)

            # Calculate error
            error = y[i] - predicted_output
            total_error += error ** 2

            # Update weights
            W0 += learning_rate * error
            W1 += learning_rate * error * X[i, 0]
            W2 += learning_rate * error * X[i, 1]

        error_list.append(total_error)
        epochs += 1

        # Check for convergence
        if total_error <= convergence_error:
            break

    return epochs, error_list

# Training data for XOR gate
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target output for XOR gate
y_xor = np.array([-1, 1, 1, -1])  # XOR gate truth table

# Learning rates to test
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Results storage
results = {}

# Iterate over learning rates
for learning_rate in learning_rates:
    epochs, error_list = train_perceptron(X_xor, y_xor, step_function, learning_rate)
    results[learning_rate] = {'epochs': epochs, 'error_list': error_list}

# Plotting the results
plt.figure(figsize=(10, 6))
for learning_rate, result in results.items():
    plt.plot(result['error_list'], label=f'Learning Rate: {learning_rate}')

plt.xlabel('Epochs')
plt.ylabel('Sum-Square Error')
plt.title('Learning Curve for XOR Gate with Different Learning Rates')
plt.legend()
plt.show()
