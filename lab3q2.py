import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean, variance

# Load the Excel file
excel_file = pd.ExcelFile("C:/Users/91630/Downloads/Lab Session1 Data.xlsx")

df = excel_file.parse('IRCTC Stock Price')

# Calculate mean and variance of the Price data in column D
price_mean = mean(df['Price'])
price_variance = variance(df['Price'])

print(f"Mean of Price: {price_mean}")
print(f"Variance of Price: {price_variance}")

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Select price data for all Wednesdays and calculate sample mean
wednesday_data = df[df['Date'].dt.day_name() == 'Wednesday']['Price']
wednesday_mean = mean(wednesday_data)

print(f"Sample mean for Wednesdays: {wednesday_mean}")

# Select price data for the month of April and calculate sample mean
april_data = df[df['Date'].dt.month == 4]['Price']
april_mean = mean(april_data)

print(f"Sample mean for April: {april_mean}")

# Probability of making a loss over the stock (Price decrease)
loss_probability = len(df[df['Price'] < df['Price'].shift(1)]) / len(df['Price'])
print(f"Probability of making a loss: {loss_probability}")

# Probability of making a profit on Wednesday
wednesday_profit_probability = len(wednesday_data[wednesday_data > wednesday_data.shift(1)]) / len(wednesday_data)
print(f"Probability of making a profit on Wednesday: {wednesday_profit_probability}")

# Conditional probability of making profit, given that today is Wednesday
conditional_profit_probability = len(wednesday_data[wednesday_data > wednesday_data.shift(1)]) / len(wednesday_data)
print(f"Conditional probability of making profit on Wednesday: {conditional_profit_probability}")

# Scatter plot of Price data against the day of the week
plt.scatter(df['Date'].dt.dayofweek, df['Price'])
plt.xlabel('Day of the Week')
plt.ylabel('Price')
plt.title('Scatter Plot of Price against Day of the Week')
plt.show()
