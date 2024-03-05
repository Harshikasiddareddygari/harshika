import numpy as np
import matplotlib.pyplot as plt
def step_activation(x):
    return 1 if x >= 0 else 0

def train_perceptron(inputs, labels, weights, learning_rate, epochs):
    errors = []

    for epoch in range(epochs):
        total_error = 0

        for i in range(len(inputs)):
            input_vector = np.insert(inputs[i], 0, 1)  
            label = labels[i]
        
            weighted_sum = np.dot(input_vector, weights)
            predicted_label = step_activation(weighted_sum)


            weights = weights + learning_rate * (label - predicted_label) * input_vector
            total_error += (label - predicted_label) ** 2
        errors.append(total_error)

    return errors, weights
initial_weights = np.array([10, 0.2, -0.75])
learning_rate = 0.05

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])

epochs = 100
error_values, final_weights = train_perceptron(inputs, labels, initial_weights, learning_rate, epochs)


plt.plot(range(1, epochs + 1), error_values, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Perceptron Training for AND Gate')
plt.grid(True)
plt.show()


print("Final Weights:", final_weights)
