import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load datasets (replace with actual paths)
diabetes_data = pd.read_csv(r'D:\AICTE-Internship\Disease-prediction-model\Data\diabetes.csv')  
heart_data = pd.read_csv(r'D:\AICTE-Internship\Disease-prediction-model\Data\heart.csv')   

columns = [
    "name", "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
    "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "status", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

file_path = r'D:\AICTE-Internship\Disease-prediction-model\Data\parkinsons.csv'
parkinsons_data = pd.read_csv(file_path, names=columns, skiprows=1)

# ---------------------------- Diabetes Model (Random Forest) ---------------------------- #
X_diabetes = diabetes_data.drop(columns=["Outcome"])  
y_diabetes = diabetes_data["Outcome"]  

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_d, y_train_d)

y_pred_d = rf_model.predict(X_test_d)
print(f"Diabetes Model Accuracy (Random Forest): {accuracy_score(y_test_d, y_pred_d):.2f}")

# Save the diabetes model
with open("diabetes_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

# ---------------------------- Heart Disease Model (Gradient Boosting) ---------------------------- #
X_heart = heart_data.drop(columns=["target"])  # Features
y_heart = heart_data["target"]  # Target variable

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_h, y_train_h)

y_pred_h = gb_model.predict(X_test_h)
print(f"Heart Disease Model Accuracy (Gradient Boosting): {accuracy_score(y_test_h, y_pred_h):.2f}")

# Save the heart disease model
with open("heart_model.pkl", "wb") as file:
    pickle.dump(gb_model, file)

# ---------------------------- Parkinson's Model (SVM) ---------------------------- #


# Drop the "name" column since it's not needed for model training
parkinsons_data.drop(columns=["name"], inplace=True)

# Split Features and Target
X_parkinsons = parkinsons_data.drop(columns=["status"])  # Features
y_parkinsons = parkinsons_data["status"]  # Target

# Convert all data to numeric
X_parkinsons = X_parkinsons.apply(pd.to_numeric, errors='coerce')
y_parkinsons = pd.to_numeric(y_parkinsons, errors='coerce')

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_parkinsons, y_parkinsons, test_size=0.2, random_state=42)

# Train SVM Model
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Model Accuracy
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Parkinson's Model Accuracy (SVM): {accuracy:.2f}")

# Save Model
with open("parkinsons_model.pkl", "wb") as model_file:
    pickle.dump(svm_model, model_file)