import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, log_loss, \
    roc_auc_score, roc_curve, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

all_files = glob.glob("C:\\uni\\diss work\\cicddos2019/*.csv")
combined = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)  # combine all files in the above location

print(combined.columns)

# print(combined.info())
# print("\nthis is a new section\n")
# print(combined.isnull().sum())


combined = combined.dropna(subset=[' Label'])  # supposed to remove rows where there isn't a label
combined = combined.drop(
    columns=['Unnamed: 0', 'Flow ID', ' Source IP', ' Destination IP', ' Timestamp', 'SimillarHTTP', ' Inbound'])

# assigns input features to anything except the label column, which is assigned to the y(target variable)
X = combined.drop(columns=[' Label'])
y = combined[' Label']

y = (y != 'BENIGN').astype(
    int)  # converts text entry of Target column to 1 and 0. 0 for benign data, and 1 for attack data

print(combined.info())  # used in testing to check data cleaning worked
print("\n\n\n", combined.columns)
print("\n\n\n", y)  # checks the conversion from string to int has worked for the target variable column

print("\n these are the x data types\n", X.dtypes)

print("\n\n\n")
for col in X.columns:
    if np.any(np.isinf(X[col])):
        print(col)  # finds columns that are higher than infinite

X = X.replace([np.inf, -np.inf], np.nan)

float32_limit = np.finfo(np.float32).max  # this finds the exact limit of float32
X = X.clip(lower=-float32_limit, upper=float32_limit)
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# split data into 80/20 training/testing set


print("success")
# build and train
forest_classifier = RandomForestClassifier(n_estimators=20, random_state=42)
forest_classifier.fit(X_train, y_train)

# prediction
y_pred = forest_classifier.predict(X_test)
importantfeatures = pd.Series(forest_classifier.feature_importances_, index=X.columns)
print(importantfeatures.sort_values(ascending=False).head(10))
# finds how the model weights certain features, returning the top 10 to check for data leakage


# evaluate the model
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred)
ROC = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
classific_rep = classification_report(y_test, y_pred)

confusion_mat = confusion_matrix(y_test, y_pred)

print(f"precision{precision}")
print(f"recall score: {recall}")
print(f"log loss score: {logloss}")
print(f"ROC score: {ROC}")
print(f"Roc_Auc score: {roc_auc}")
print(f"accuracy: {accuracy:.2f}")
print("\n\n\n")
print(f"confusion matrix: {confusion_mat}")
print("\n\n\n")
print("\n Classification report:\n", classific_rep)
