import numpy as np
import matplotlib.pyplot as plt

# Initial weights
W0 = 10
W1 = 0.2
W2 = -0.75

# Training data for AND gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

max_epochs = 1000

learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

learning_rates_list = []
iterations_list = []

def step_function(x):
    return 1 if x >= 0 else 0

def train_perceptron(learning_rate):
    global W0, W1, W2
    iterations = 0

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
            W0 += learning_rate * error
            W1 += learning_rate * error * X[i, 0]
            W2 += learning_rate * error * X[i, 1]

        iterations += 1

        # Check for convergence
        if total_error <= 0.002:
            break

    return iterations

for lr in learning_rates:
    W0 = 10  # Reset weights for each learning rate
    W1 = 0.2
    W2 = -0.75
    iterations = train_perceptron(lr)
    
    # Save learning rate and corresponding iterations for plotting
    learning_rates_list.append(lr)
    iterations_list.append(iterations)

# Plotting
plt.plot(learning_rates_list, iterations_list, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations to Converge')
plt.title('Learning Rate vs. Convergence Iterations')
plt.show()
