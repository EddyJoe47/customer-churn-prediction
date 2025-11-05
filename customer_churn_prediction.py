import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("🔹 Loading dataset...")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
print("✅ Dataset Loaded Successfully!\n")

print("🔹 First few rows of the dataset:")
print(df.head(), "\n")

# Preprocessing
print("🔹 Preprocessing data...")
df = df.dropna()

# Convert 'Churn' column to numeric (Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Drop customerID (non-numeric)
df = df.drop('customerID', axis=1)

# Convert categorical columns to dummy variables
df = pd.get_dummies(df, drop_first=True)
print("✅ Preprocessing Completed!\n")

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split dataset
print("🔹 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data Split Done!\n")

# Train Random Forest model
print("🔹 Training Random Forest model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("✅ Model Trained Successfully!\n")

# Predict and evaluate
print("🔹 Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.2f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))
