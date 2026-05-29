# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("dataset/Titanic-Dataset.csv")

# Display First 5 Rows
print("First 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Info:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin Column (Too Many Missing Values)
df.drop(columns=['Cabin'], inplace=True)

# Verify Missing Values Again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Convert Categorical Columns into Numerical
label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])

df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# Display Encoded Data
print("\nEncoded Dataset:")
print(df.head())

# Feature Scaling
scaler = StandardScaler()

numerical_cols = ['Age', 'Fare']

df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\nScaled Dataset:")
print(df.head())

# Boxplot for Outlier Detection
plt.figure(figsize=(8,5))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot for Outlier Detection")
plt.show()

# Remove Outliers Using IQR Method
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

print("\nDataset Shape After Removing Outliers:")
print(df.shape)

# Save Cleaned Dataset
df.to_csv("dataset/cleaned_titanic.csv", index=False)

print("\nData Cleaning and Preprocessing Completed Successfully!")