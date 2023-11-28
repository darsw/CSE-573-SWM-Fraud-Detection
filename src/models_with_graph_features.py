import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
df1 = pd.read_csv("/content/drive/MyDrive/swm_project/data/graphfeatures.csv")



from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
X = df1.drop('fraud', axis=1)
y = df1['fraud']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Random Over-Sampling only on training data
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

# Create a new balanced DataFrame
df_balanced = pd.DataFrame(X_train_ros, columns=X_train.columns)
df_balanced['fraud'] = y_train_ros

df  = df_balanced
X_train_ros = X_train_ros.drop('fraud', axis=1)
best_parameters =  {'gamma': 0, 'learning_rate': 0.5, 'max_depth': 20, 'n_estimators': 300}

xgb_model_optimized = XGBClassifier(**best_parameters, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model_optimized.fit(X_train_ros, y_train_ros)
optimized_predictions = xgb_model_optimized.predict(X_test)
optimized_metrics = evaluate_model(y_test, optimized_predictions)

print("\nOptimized XGBoost Model Evaluation Metrics:")
print("Accuracy:", optimized_metrics[0])
print("Precision:", optimized_metrics[1])
print("Recall:", optimized_metrics[2])
print("F1 Score:", optimized_metrics[3])
print("Confusion Matrix:\n", optimized_metrics[4])



# Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train_ros, y_train_ros)
rf_predictions = rf_model.predict(X_test)

# Function to calculate evaluation metrics
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, confusion

# Evaluating SVM Model


rf_metrics = evaluate_model(y_test, rf_predictions)

print("\nRandom Forest Model Evaluation Metrics:")
print("Accuracy:", rf_metrics[0])
print("Precision:", rf_metrics[1])
print("Recall:", rf_metrics[2])
print("F1 Score:", rf_metrics[3])
print("Confusion Matrix:\n", rf_metrics[4])
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Assuming you have already split your data into X_train, X_test, y_train, and y_test

# Initialize the Logistic Regression model
log_reg_model = LogisticRegression()

# Fit the model to the training data
log_reg_model.fit(X_train_ros, y_train_ros)

# Predict on the test data
log_reg_predictions = log_reg_model.predict(X_test)

# Function to calculate evaluation metrics
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, confusion

# Evaluating Logistic Regression Model
log_reg_metrics = evaluate_model(y_test, log_reg_predictions)

print("\nLogistic Regression Model Evaluation Metrics:")
print("Accuracy:", log_reg_metrics[0])
print("Precision:", log_reg_metrics[1])
print("Recall:", log_reg_metrics[2])
print("F1 Score:", log_reg_metrics[3])
print("Confusion Matrix:\n", log_reg_metrics[4])


# # Initialize the SVM model
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# svm_model = SVC()

# svm_model.fit(X_train_ros, y_train_ros)

# svm_predictions = svm_model.predict(X_test)

# def evaluate_model(y_true, y_pred):
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     confusion = confusion_matrix(y_true, y_pred)
#     return accuracy, precision, recall, f1, confusion

# svm_metrics = evaluate_model(y_test, svm_predictions)

# print("\nSVM Model Evaluation Metrics after PCA:")
# print("Accuracy:", svm_metrics[0])
# print("Precision:", svm_metrics[1])
# print("Recall:", svm_metrics[2])
# print("F1 Score:", svm_metrics[3])
# print("Confusion Matrix:\n", svm_metrics[4])