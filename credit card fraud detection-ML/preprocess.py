import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
data = pd.read_csv('fraudTest.csv')

# Drop unnecessary columns
data.drop(columns=['Unnamed: 0', 'trans_num', 'dob'], inplace=True)

# Convert 'trans_date_trans_time' to datetime
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])

# Extract date features
data['trans_hour'] = data['trans_date_trans_time'].dt.hour
data['trans_minute'] = data['trans_date_trans_time'].dt.minute
data['trans_second'] = data['trans_date_trans_time'].dt.second

# Drop the original datetime column
data.drop(columns=['trans_date_trans_time'], inplace=True)

# Identify categorical columns
categorical_columns = ['merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'job']

# Encode categorical variables using Label Encoding
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separate features and target variable
X = data.drop(columns=['is_fraud'])
y = data['is_fraud']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the data types and unique values before scaling
print("Data Types in X_train:")
print(X_train.dtypes)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the preprocessed data
pd.DataFrame(X_train, columns=X.columns).to_csv('X_train.csv', index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv('X_test.csv', index=False)

print("Data preprocessing complete. Train and test sets saved.")
