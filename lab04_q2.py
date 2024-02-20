import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read datasets using Pandas
data1 = pd.read_csv('C:/Users/91630/Downloads/code_ques.csv')
data2 = pd.read_csv('C:/Users/91630/Downloads/code_sol.csv')
data3 = pd.read_csv('C:/Users/91630/Downloads/code_comm.csv')
data4 = pd.read_csv('C:/Users/91630/Downloads/code_only.csv')

# Extract the entire dataset for each file
feature_data1 = data1.to_numpy().flatten()
feature_data2 = data2.to_numpy().flatten()
feature_data3 = data3.to_numpy().flatten()
feature_data4 = data4.to_numpy().flatten()

# Plotting histograms for all four datasets
plt.hist(feature_data1[~np.isnan(feature_data1)], bins=10, color='blue', alpha=0.5, label='Dataset 1', edgecolor='black')
plt.hist(feature_data2[~np.isnan(feature_data2)], bins=10, color='red', alpha=0.5, label='Dataset 2', edgecolor='black')
plt.hist(feature_data3[~np.isnan(feature_data3)], bins=10, color='green', alpha=0.5, label='Dataset 3', edgecolor='black')
plt.hist(feature_data4[~np.isnan(feature_data4)], bins=10, color='purple', alpha=0.5, label='Dataset 4', edgecolor='black')

plt.title('Histogram Comparison of Your Feature')
plt.xlabel('Feature Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Calculate mean and variance for all four datasets
mean_value1, variance_value1 = np.nanmean(feature_data1), np.nanvar(feature_data1)
mean_value2, variance_value2 = np.nanmean(feature_data2), np.nanvar(feature_data2)
mean_value3, variance_value3 = np.nanmean(feature_data3), np.nanvar(feature_data3)
mean_value4, variance_value4 = np.nanmean(feature_data4), np.nanvar(feature_data4)

# Print mean and variance values
print(f'Dataset 1 - Mean: {round(mean_value1, 2)}, Variance: {round(variance_value1, 2)}')
print(f'Dataset 2 - Mean: {round(mean_value2, 2)}, Variance: {round(variance_value2, 2)}')
print(f'Dataset 3 - Mean: {round(mean_value3, 2)}, Variance: {round(variance_value3, 2)}')
print(f'Dataset 4 - Mean: {round(mean_value4, 2)}, Variance: {round(variance_value4, 2)}')
