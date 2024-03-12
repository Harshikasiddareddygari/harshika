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
    hidden_size = 4  #  adjust the number of hidden units
    output_size = 2  # Updated for two output nodes

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

# Training data for AND gate
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])  # Updated for two output nodes

# Hyperparameters
learning_rate_and = 0.05
epochs_and = 1000
convergence_error_and = 0.002

# Train the neural network for AND gate logic
weights_input_hidden_and, weights_hidden_output_and, bias_hidden_and, bias_output_and, error_list_and = backpropagation(X_and, y_and, learning_rate_and, epochs_and, convergence_error_and)

# Test the neural network
hidden_layer_input_and = np.dot(X_and, weights_input_hidden_and) + bias_hidden_and
hidden_layer_output_and = sigmoid(hidden_layer_input_and)

output_layer_input_and = np.dot(hidden_layer_output_and, weights_hidden_output_and) + bias_output_and
predicted_output_and = sigmoid(output_layer_input_and)

# Print the final predictions for AND gate logic
print("Final Predictions for AND Gate Logic with Two Output Nodes:")
print(predicted_output_and)
