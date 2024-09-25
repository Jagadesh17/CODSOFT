import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load preprocessed test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('fraudTest.csv')['is_fraud'][len(pd.read_csv('X_train.csv')):]  # Aligning lengths

# Load the trained model
model = joblib.load('model/model.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
