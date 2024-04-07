import pandas as pd
import numpy as np

def bin_continuous_feature(data, feature_name='score', num_bins=None, binning_type='equal_width'):
    # Check if num_bins is provided, otherwise use a default value
    if num_bins is None:
        num_bins = 10  # Default number of bins
    
    # Check binning_type and perform binning accordingly
    if binning_type == 'equal_width':
        bins = pd.cut(data[feature_name], bins=num_bins, labels=False)
    elif binning_type == 'frequency':
        bins = pd.qcut(data[feature_name], q=num_bins, labels=False, duplicates='drop')
    else:
        raise ValueError("Invalid binning_type. Choose 'equal_width' or 'frequency'.")

    # Add bin column to the dataframe
    data['bin_' + feature_name] = bins

    return data

# Example usage:
data = pd.read_csv("C:/Users/91630/Downloads/code_only (1).csv")  # Load your dataset
data = data.dropna()  # Drop rows with NaN values

# Perform equal width binning with 5 bins on the 'score' column
data_binned_equal_width = bin_continuous_feature(data, feature_name='score', num_bins=5, binning_type='equal_width')

# Perform frequency binning with 4 bins on the 'score' column
data_binned_frequency = bin_continuous_feature(data, feature_name='score', num_bins=4, binning_type='frequency')

# Print the modified dataframe with bin columns added
print(data_binned_equal_width.head())
print(data_binned_frequency.head())
