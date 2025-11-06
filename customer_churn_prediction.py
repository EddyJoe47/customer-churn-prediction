import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print(" Starting Customer Churn Prediction...\n")

print("Loading dataset...")
url = "https://raw.githubusercontent.com/EddyJoe47/customer-churn-prediction/refs/heads/main/customer_churn_prediction.csv"
df = pd.read_csv(url)
print(" Dataset Loaded Successfully!")
print(f" Shape of dataset: {df.shape}\n")

print(" Displaying first 5 rows:")
print(df.head(), "\n")

print(" Preprocessing data...")

# Drop missing values
df = df.dropna()

# Convert 'Churn' column to numeric (Yes = 1, No = 0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Drop non-numeric 'customerID' column
df = df.drop('customerID', axis=1)

# Convert categorical variables to numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

print(" Preprocessing Completed!")
print(f" New dataset shape after encoding: {df.shape}\n")

X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data into training and testing sets
print(" Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(" Data Split Done!\n")

print(" Training Random Forest model...")
model = RandomForestClassifier(random_state=42, n_estimators=200)
model.fit(X_train, y_train)
print(" Model Trained Successfully!\n")

print(" Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f" Model Accuracy: {accuracy:.2%}\n")
print(" Classification Report:")
print(classification_report(y_test, y_pred))

import numpy as np
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

print("\n Top 10 Important Features:")
print(top_features)
