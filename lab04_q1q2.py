# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data from two CSV files into DataFrames
class1_data = pd.read_csv('C:/Users/91630/Downloads/code_comm.csv')
class2_data = pd.read_csv('C:/Users/91630/Downloads/code_only.csv')

# Extract vectors from DataFrames
class1_vectors = class1_data['0'].values
class2_vectors = class2_data['0'].values

# Convert vectors to NumPy arrays
class1_vectors = np.array(class1_vectors)
class2_vectors = np.array(class2_vectors)

# Calculate centroids for both classes
centroid_class1 = np.mean(class1_vectors, axis=0)
centroid_class2 = np.mean(class2_vectors, axis=0)

# Calculate spreads (standard deviations) for both classes
spread_class1 = np.std(class1_vectors, axis=0)
spread_class2 = np.std(class2_vectors, axis=0)

# Calculate Euclidean distance between centroids
distance_between_centroids = np.linalg.norm(centroid_class1 - centroid_class2)

# Display results
print("Centroid of class 1:", centroid_class1)
print("Centroid of class 2:", centroid_class2)
print("Spread of class 1:", spread_class1)
print("Spread of class 2:", spread_class2)
print("Distance between centroids:", distance_between_centroids)

# Select a feature for analysis
selected_feature = '0'  

# Concatenate vectors for histogram plotting
feature_data = np.concatenate((class1_vectors, class2_vectors))

# Set the number of bins for the histogram
num_bins = 20  

# Generate histograms for both classes
hist, bins = np.histogram(feature_data, bins=num_bins)
plt.hist(class1_vectors, bins=num_bins, alpha=0.5, label='Class 1')
plt.hist(class2_vectors, bins=num_bins, alpha=0.5, label='Class 2')

# Plot labels and title
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.title('Histogram of Feature: {}'.format(selected_feature))
plt.legend()
plt.show()

# Calculate mean and variance of the selected feature
mean_value = np.mean(feature_data)
variance_value = np.var(feature_data)

# Display mean and variance results
print("Mean of the feature:", mean_value)
print("Variance of the feature:", variance_value)
