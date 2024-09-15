import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the HR Analytics dataset
hr_df = pd.read_csv(r'aug_test.csv')

# Data Preprocessing: Drop rows with missing values
hr_df.dropna(inplace=True)

# Define target and features
y = hr_df['relevent_experience'].apply(lambda x: 1 if x == 'Has relevent experience' else 0)  # Binary target
X = hr_df.drop(columns=['enrollee_id', 'relevent_experience'])  # Drop target and irrelevant columns

# One-Hot Encoding for Categorical Columns
X_encoded = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical features

# Sensitive feature for bias detection (e.g., 'gender')
sensitive_feature = hr_df['gender'].apply(lambda x: 1 if x == 'Female' else 0)

# Train/Test Split
X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X_encoded, y, sensitive_feature, test_size=0.3, random_state=42
)

# Scale Features (only numerical features need scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: \n{conf_matrix}')
print(f'ROC AUC Score: {roc_auc}')

# ------------------ Visualizations ------------------

# 1. Distribution of Relevant Experience (Target)
sns.countplot(x='relevent_experience', data=hr_df)
plt.title('Distribution of Relevant Experience')
plt.xlabel('Relevant Experience')
plt.ylabel('Count')
plt.show()

# 2. Plot Confusion Matrix
fig, ax = plt.subplots()
cax = ax.matshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 3. ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'Logistic Regression (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# 4. Bias Evaluation: Mean Prediction for Male vs Female
group_0_pred_mean = np.mean(y_pred[sensitive_test == 0])  # Male
group_1_pred_mean = np.mean(y_pred[sensitive_test == 1])  # Female

# Bar plot to show parity between genders
parity_df = pd.DataFrame({
    'Group': ['Male', 'Female'],
    'Mean Prediction': [group_0_pred_mean, group_1_pred_mean]
})

sns.barplot(x='Group', y='Mean Prediction', data=parity_df)
plt.title('Mean Prediction for Gender Groups')
plt.ylabel('Mean Predicted Probability')
plt.show()
