import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def bipolar_step_function(x):
    return 1 if x >= 0 else -1

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
    return max(0, x)

# Perceptron training function
def train_perceptron(inputs, targets, weights, activation_function, learning_rate, max_epochs=1000):
    error_values = []

    for epoch in range(max_epochs):
        total_error = 0

        for i in range(len(inputs)):
            input_vector = np.insert(inputs[i], 0, 1)  # Add bias term
            output = activation_function(np.dot(weights, input_vector))
            error = targets[i] - output
            total_error += error**2
            weights += learning_rate * error * input_vector

        error_values.append(total_error)

        if total_error == 0:
            print(f"Convergence reached in {epoch + 1} epochs")
            break  # Convergence reached

    return weights, error_values

# AND gate training data
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

targets = np.array([-1, -1, -1, 1])  # AND gate logic

# Initial weights and parameters
initial_weights = np.array([10, 0.2, -0.75])
learning_rate = 0.05

# Train perceptron with Bi-Polar Step function
weights_bipolar, errors_bipolar = train_perceptron(inputs, targets, initial_weights.copy(),
                                                   bipolar_step_function, learning_rate)
# Plotting
plt.plot(range(1, len(errors_bipolar) + 1), errors_bipolar, label='Bi-Polar Step')

plt.xlabel('Epochs')
plt.ylabel('Sum Square Error')
plt.title('Convergence with Bi-Polar Step Function')
plt.legend()
plt.show()

# Train perceptron with Sigmoid function
weights_sigmoid, errors_sigmoid = train_perceptron(inputs, targets, initial_weights.copy(),
                                                   sigmoid_function, learning_rate)
# Plotting
plt.plot(range(1, len(errors_sigmoid) + 1), errors_sigmoid, label='Sigmoid')

plt.xlabel('Epochs')
plt.ylabel('Sum Square Error')
plt.title('Convergence with Sigmoid Function')
plt.legend()
plt.show()

# Train perceptron with ReLU function
weights_relu, errors_relu = train_perceptron(inputs, targets, initial_weights.copy(),
                                             relu_function, learning_rate)
# Plotting
plt.plot(range(1, len(errors_relu) + 1), errors_relu, label='ReLU')

plt.xlabel('Epochs')
plt.ylabel('Sum Square Error')
plt.title('Convergence with ReLU Function')
plt.legend()
plt.show()

# Print epochs for each activation function
print(f"Epochs for Bi-Polar Step: {len(errors_bipolar)}")
print(f"Epochs for Sigmoid: {len(errors_sigmoid)}")
print(f"Epochs for ReLU: {len(errors_relu)}")
