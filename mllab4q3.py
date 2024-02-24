import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

# Function to calculate Minkowski distance
def calculate_minkowski_distance(vector1, vector2, r):
    return minkowski(vector1, vector2, r)

# Load your datasets using pandas
# Assuming your datasets are in CSV files, modify the file paths accordingly
file_path1 = 'C:/Users/91630/Downloads/code_comm.csv'
file_path2 = 'C:/Users/91630/Downloads/code_only.csv'
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Choose two rows as feature vectors (modify index values accordingly)
vector1 = df1.iloc[0, 1:]  # Assuming the features start from the second column
vector2 = df2.iloc[1, 1:]

# Range of r values
r_values = range(1, 11)

# Calculate distances for each r
distances = [calculate_minkowski_distance(vector1, vector2, r) for r in r_values]

# Plot the distances
plt.plot(r_values, distances, marker='o')
plt.title('Minkowski Distance vs r')
plt.xlabel('r')
plt.ylabel('Distance')
plt.grid(True)
plt.show()
