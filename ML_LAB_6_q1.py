import numpy as np
import matplotlib.pyplot as plt

# Initial weights
W0 = 10
W1 = 0.2
W2 = -0.75

# Learning rate
alpha = 0.05

# Training data for AND gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target labels for AND gate
y = np.array([0, 0, 0, 1])

# Number of epochs
max_epochs = 1000

# Lists to store epoch and error values for plotting
epochs_list = []
error_list = []

# Step activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Training the perceptron
for epoch in range(max_epochs):
    total_error = 0
    for i in range(len(X)):
        # Forward pass
        net_input = W0 + W1 * X[i, 0] + W2 * X[i, 1]
        predicted_output = step_function(net_input)

        # Calculate error
        error = y[i] - predicted_output
        total_error += error ** 2

        # Update weights
        W0 += alpha * error
        W1 += alpha * error * X[i, 0]
        W2 += alpha * error * X[i, 1]

    # Save epoch and error for plotting
    epochs_list.append(epoch)
    error_list.append(total_error)

    # Check for convergence
    if total_error <= 0.002:
        print(f"Converged in {epoch+1} epochs.")
        break
# Display final weights
print("Final Weights:")
print(f"W0 = {W0}, W1 = {W1}, W2 = {W2}")

# Plotting
plt.plot(epochs_list, error_list)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square Error')
plt.title('Perceptron Learning Curve')
plt.show()


