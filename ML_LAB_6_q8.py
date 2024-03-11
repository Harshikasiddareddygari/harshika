import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the Sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Function to initialize weights and biases
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    bias_hidden = np.zeros((1, hidden_size))
    bias_output = np.zeros((1, output_size))

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Backpropagation algorithm
def backpropagation(X, y, learning_rate, epochs, convergence_error):
    input_size = X.shape[1]
    hidden_size = 4  # You can adjust the number of hidden units
    output_size = 1

    # Initialize weights and biases
    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = initialize_parameters(input_size, hidden_size, output_size)

    error_list = []

    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        predicted_output = sigmoid(output_layer_input)

        # Calculate error
        error = y - predicted_output
        error_list.append(np.mean(np.abs(error)))

        # Check for convergence
        if np.mean(np.abs(error)) <= convergence_error:
            print(f"Converged in {epoch+1} epochs.")
            break

        # Backward pass
        output_error = error * sigmoid_derivative(predicted_output)
        hidden_layer_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

        # Update weights and biases
        weights_hidden_output += learning_rate * hidden_layer_output.T.dot(output_error)
        weights_input_hidden += learning_rate * X.T.dot(hidden_layer_error)
        bias_output += learning_rate * np.sum(output_error, axis=0, keepdims=True)
        bias_hidden += learning_rate * np.sum(hidden_layer_error, axis=0, keepdims=True)

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, error_list

# Training data for XOR gate
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# Hyperparameters
learning_rate_xor = 0.05
epochs_xor = 1000
convergence_error_xor = 0.002

# Train the neural network for XOR gate logic
weights_input_hidden_xor, weights_hidden_output_xor, bias_hidden_xor, bias_output_xor, error_list_xor = backpropagation(X_xor, y_xor, learning_rate_xor, epochs_xor, convergence_error_xor)

# Test the neural network
hidden_layer_input_xor = np.dot(X_xor, weights_input_hidden_xor) + bias_hidden_xor
hidden_layer_output_xor = sigmoid(hidden_layer_input_xor)

output_layer_input_xor = np.dot(hidden_layer_output_xor, weights_hidden_output_xor) + bias_output_xor
predicted_output_xor = sigmoid(output_layer_input_xor)

# Print the final predictions for XOR gate logic
print("Final Predictions for XOR Gate Logic:")
print(predicted_output_xor)
