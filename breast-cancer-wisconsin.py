#First we load and pre-process the data to make it ready for training
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('wdbc.data', header=None)

# Load feature names from wdbc.names
with open('wdbc.names', 'r') as f:
    names = f.read().split('\n')[5:35]
    feature_names = [name.split(":")[0] for name in names]
feature_names = ['ID', 'Diagnosis'] + feature_names

data.columns = feature_names

# Drop the ID column
data.drop(columns=['ID'], inplace=True)

# Map diagnosis to numeric
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

# Split the data into features and target
X = data.drop(columns=['Diagnosis'])
y = data['Diagnosis']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#First ML Model Logistic Regression
from sklearn.linear_model import LogisticRegression

# Initialize and train the model
logistic_model = LogisticRegression(max_iter=10000)
logistic_model.fit(X_train, y_train)

# Make predictions
y_pred_logistic = logistic_model.predict(X_test)

# Evaluate the model
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f'Logistic Regression Accuracy: {accuracy_logistic}')
