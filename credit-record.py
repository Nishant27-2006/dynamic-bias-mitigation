import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Credit Record dataset
credit_record_df = pd.read_csv(r'credit_record.csv')

# Binarize 'STATUS' column: 0 for good credit (e.g., '0' or 'C'), 1 for bad credit (e.g., '1', '2', etc.)
credit_record_df['STATUS_BINARY'] = credit_record_df['STATUS'].apply(lambda x: 1 if x in ['1', '2', '3', '4', '5'] else 0)

# Drop irrelevant columns
X = credit_record_df.drop(columns=['ID', 'MONTHS_BALANCE', 'STATUS'])

# Target column
y = credit_record_df['STATUS_BINARY']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:,1]

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: \n{conf_matrix}')
print(f'ROC AUC Score: {roc_auc}')

# ------------------ Visualizations ------------------

# 1. Distribution of Good vs Bad Credit Status
sns.countplot(x='STATUS_BINARY', data=credit_record_df)
plt.title('Distribution of Good vs Bad Credit Status')
plt.xlabel('Credit Status (0 = Good, 1 = Bad)')
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
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'Logistic Regression (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# 4. Bias Over Time (if applicable)
if 'MONTHS_BALANCE' in credit_record_df.columns:
    monthly_bias = credit_record_df.groupby('MONTHS_BALANCE')['STATUS_BINARY'].mean()
    plt.plot(monthly_bias.index, monthly_bias.values)
    plt.title('Bias Over Time')
    plt.xlabel('Months Balance')
    plt.ylabel('Mean Status (1 = Bad Credit)')
    plt.show()

# 5. Fairness Evaluation (Parity between Sensitive Groups)
# Assuming sensitive feature is gender, replace 'gender' with actual sensitive column
if 'gender' in credit_record_df.columns:
    group_0_pred_mean = np.mean(y_pred[credit_record_df['gender'][X_test.index] == 0])
    group_1_pred_mean = np.mean(y_pred[credit_record_df['gender'][X_test.index] == 1])

    # Bar plot to show parity between groups
    parity_df = pd.DataFrame({
        'Group': ['Group 0', 'Group 1'],
        'Mean Prediction': [group_0_pred_mean, group_1_pred_mean]
    })

    sns.barplot(x='Group', y='Mean Prediction', data=parity_df)
    plt.title('Mean Prediction for Sensitive Groups')
    plt.ylabel('Mean Predicted Probability')
    plt.show()
