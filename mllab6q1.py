import numpy as np
import matplotlib.pyplot as plt

# Step activation function: Outputs 1 if x is greater than or equal to 0, else 0
def step_activation(x):
    return 1 if x >= 0 else 0

# Training the perceptron
def train_perceptron(inputs, labels, weights, learning_rate, epochs):
    errors = []

    # Loop through epochs
    for epoch in range(epochs):
        total_error = 0

        # Loop through input data
        for i in range(len(inputs)):
            # Insert bias (1) into the input vector
            input_vector = np.insert(inputs[i], 0, 1)  
            label = labels[i]
        
            # Calculate weighted sum and predict output
            weighted_sum = np.dot(input_vector, weights)
            predicted_label = step_activation(weighted_sum)

            # Update weights based on the learning algorithm
            weights = weights + learning_rate * (label - predicted_label) * input_vector
            total_error += (label - predicted_label) ** 2
        
        # Store total error for the epoch
        errors.append(total_error)

    return errors, weights

# Initial weights and learning rate
initial_weights = np.array([10, 0.2, -0.75])
learning_rate = 0.05

# Input data and labels for AND gate
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])

# Number of training epochs
epochs = 100

# Train the perceptron and get error values and final weights
error_values, final_weights = train_perceptron(inputs, labels, initial_weights, learning_rate, epochs)

# Plotting the sum-square-error over epochs
plt.plot(range(1, epochs + 1), error_values, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Perceptron Training for AND Gate')
plt.grid(True)
plt.show()

# Display the final weights after training
print("Final Weights:", final_weights)

