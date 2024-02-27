import pandas as pd
import matplotlib.pyplot as plt

# Load the first dataset
file_path1 = 'C:/Users/91630/Downloads/code_comm.csv'  # Update the path to your first file
df1 = pd.read_csv(file_path1, header=None, skiprows=1, dtype=float)

# Load the second dataset
file_path2 = 'C:/Users/91630/Downloads/code_only.csv'  # Update the path to your second file
df2 = pd.read_csv(file_path2, header=None, skiprows=1, dtype=float)

# Extract features X and Y for the first dataset
X1 = df1.iloc[:, 0]
Y1 = df1.iloc[:, 1]
class_labels1 = df1.iloc[:, 2]

# Extract features X and Y for the second dataset
X2 = df2.iloc[:, 0]
Y2 = df2.iloc[:, 1]
class_labels2 = df2.iloc[:, 2]

# Create subplots for each dataset
plt.figure(figsize=(12, 5))

# Plot for the first dataset
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X1, Y1, c=class_labels1, cmap='bwr', marker='o')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.title('Scatter Plot - Dataset 1')
cb1 = plt.colorbar(scatter1, ticks=[0, 1])
cb1.set_ticklabels(['Blue', 'Red'])

# Plot for the second dataset
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X2, Y2, c=class_labels2, cmap='bwr', marker='o')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.title('Scatter Plot - Dataset 2')
cb2 = plt.colorbar(scatter2, ticks=[0, 1])
cb2.set_ticklabels(['Blue', 'Red'])

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
