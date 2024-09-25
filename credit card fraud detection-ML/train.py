import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load preprocessed datasets
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('fraudTest.csv')['is_fraud'][:len(X_train)]  # Aligning lengths
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('fraudTest.csv')['is_fraud'][len(X_train):]  # Aligning lengths

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model/model.pkl')

print("Model trained and saved successfully!")
