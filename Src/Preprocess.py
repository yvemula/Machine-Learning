import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('titanic.csv')

# Display basic information
print(data.info())
print(data.describe())

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(columns='Cabin', inplace=True)

# Drop irrelevant columns
data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

# Convert categorical variables into numeric
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Split the data into features and target variable
X = data.drop(columns='Survived')
y = data['Survived']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
