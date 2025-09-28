#Task 4: Classification with Logistic Regression.
#1.Choose a binary classification dataset.
#2.Train/test split and standardize features.
#3.Fit a Logistic Regression model.
#4.Evaluate with confusion matrix, precision, recall, ROC-AUC.
#5.Tune threshold and explain sigmoid function.

# Logistic Regression Binary Classifier (Housing Dataset)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 1. Load and prepare datase
df = pd.read_csv("Housing.csv")

# Create binary target: expensive (>= median price)
median_price = df["price"].median()
df["expensive"] = (df["price"] >= median_price).astype(int)

# Drop original price column (since we now use binary target)
df = df.drop(columns=["price"])

# Encode categorical variables
categorical_cols = df.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 2. Train-test split

X = df.drop("expensive", axis=1)
y = df["expensive"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Fit Logistic Regression

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 4. Evaluation

y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Precision, Recall, F1-score
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("ROC-AUC Score:", roc_auc)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()

# 5. Threshold Tuning

threshold = 0.6  # Example: classify as expensive if prob >= 0.6
y_pred_custom = (y_pred_prob >= threshold).astype(int)
cm_custom = confusion_matrix(y_test, y_pred_custom)
print(f"\nConfusion Matrix with threshold={threshold}:\n", cm_custom)

# 6. Sigmoid Function Explanation

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z))
plt.title("Sigmoid Function")
plt.xlabel("z (linear combination of features)")
plt.ylabel("Sigmoid(z)")
plt.grid(True)
plt.show()



