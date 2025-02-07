# These Models were trained on Linear regression 



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# Load dataset with corrected file path
parkinsons_dataset = pd.read_csv(r'D:\AICTE-Internship\Disease-prediction-model\Data\parkinsons.csv')

# Display the first few rows
print(parkinsons_dataset.head())

# Dataset shape
print("Dataset shape:", parkinsons_dataset.shape)

# Dataset statistics
print(parkinsons_dataset.describe())

# Count of target variable classes
print(parkinsons_dataset['status'].value_counts())

# Grouped statistics by 'status' (numeric columns only)
numeric_columns = parkinsons_dataset.select_dtypes(include=[np.number])
grouped_means = numeric_columns.groupby(parkinsons_dataset['status']).mean()
print(grouped_means)

# Splitting data into features and target
x = parkinsons_dataset.drop(columns=['name', 'status'], axis=1)  # Drop non-numeric columns
y = parkinsons_dataset['status']
print(x.head())

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)

# Creating and training the SVM model
model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)

# Making predictions
y_pred = model.predict(x_test)

# Calculating accuracy
print('Accuracy:', accuracy_score(y_test, y_pred))

# Saving the trained model
filename = 'parkinson-prediction-model.pkl'
pickle.dump(model, open(filename, 'wb'))
