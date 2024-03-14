#importing necessary libraries

import numpy as np
import pandas as pd
import os
#File Path
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

url = r'C:\Users\arivu\Downloads\Transac\Hackathon_Working_Data.csv'#Repository Location
trans_data = pd.read_csv(url)

#Dataset Overview
print("Dataset Overview:")
print(trans_data.info())

#Printing first two rows of the dataset
print("\nFirst Few Rows of the Dataset:")
print(trans_data.head())
print("\nSummary Statistics:")
print(trans_data.describe())

#Summary statistics
print("\nSummary Statistics:")
print(trans_data.describe())

#Handling dataset
trans_data = trans_data.drop_duplicates()
numeric_columns = trans_data.select_dtypes(include=['number']).columns
trans_data[numeric_columns] = trans_data[numeric_columns].fillna(trans_data[numeric_columns].mean())
categorical_columns = trans_data.select_dtypes(exclude=['number']).columns
trans_data[categorical_columns] = trans_data[categorical_columns].fillna('Unknown')

#Data Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='DAY', data=trans_data)
plt.xlabel("Day")
plt.ylabel("Frequency")
plt.title('DAY Distribution')
plt.show()

import seaborn as sns
plt.figure(figsize=(10, 6))
sns.histplot(trans_data['PRICE'], bins=90, kde=True)  
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

sns.scatterplot(x='Total Spend', y='DAY', data=trans_data)
plt.title('Total Spend vs. Day of Transaction')
plt.xlabel('Total Spend')
plt.ylabel('Day of Transaction')
plt.show()

#Correlation matrix
numeric_columns = trans_data.select_dtypes(include='number')
correlation_matrix = numeric_columns.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()








