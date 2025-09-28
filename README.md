# housing-logistic-regression
Binary classification of housing prices using logistic regression. Includes preprocessing, model training, evaluation (confusion matrix, ROC-AUC), threshold tuning, and sigmoid explanation.

# Logistic Regression Binary Classifier - Housing Dataset

## 📌 Objective
Build a **binary classifier** using **Logistic Regression** to predict whether a house is **expensive (1)** or **not expensive (0)** based on housing features.

---

## 🛠 Tools Used
- **Python**
- **Pandas** → data handling
- **Scikit-learn** → train-test split, preprocessing, logistic regression, evaluation
- **Matplotlib** → plotting (ROC curve, Sigmoid function)

---

## 📂 Dataset
- File: `Housing.csv`
- 545 rows × 13 columns
- Features: `area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus`
- Target: Converted `price` into binary `expensive` (1 = above/equal to median price, 0 = below median).

---

## 🚀 Steps
1. **Preprocessing**
   - Encoded categorical variables using `LabelEncoder`.
   - Scaled numerical features using `StandardScaler`.

2. **Model Training**
   - Used **Logistic Regression**.

3. **Evaluation**
   - Confusion Matrix
   - Precision, Recall, F1-score
   - ROC Curve & AUC score

4. **Threshold Tuning**
   - Changed classification threshold from default `0.5` to custom `0.6` to see effect.

5. **Sigmoid Function**
   - Explained how logistic regression converts linear values into probabilities between 0 and 1.

---

## 📊 Results
- **Confusion Matrix**: Shows TP, TN, FP, FN counts.
- **Classification Report**: Precision, Recall, F1-score.
- **ROC-AUC**: Evaluates classifier’s ability to separate classes.
- **Sigmoid Curve**: Demonstrates probability mapping.

---

## 🔑 Key Takeaways
- Logistic Regression works well for binary classification.
- Threshold tuning can balance **precision vs recall**.
- ROC-AUC gives a robust measure of performance.
- Sigmoid maps feature combinations to probabilities for decision-making.
