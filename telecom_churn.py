# AUTHOR: UBANDIYA Najib Yusuf

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd

try:
    df1 = pd.read_csv('telecom_demographics.csv')
    df2 = pd.read_csv('telecom_usage.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

if not df1.empty and not df2.empty:
    churn_df = pd.merge(df1, df2)

    churn_rate = churn_df['churn'].mean()

    categorical_vars = churn_df.select_dtypes(include=['object', 'category']).columns.tolist()

    churn_rate, categorical_vars
else:
    print("One or both of the DataFrames are empty. Please check the CSV files.")

# EDA
print(churn_df.head())
print()

print(churn_df.describe(include='all'))
print()

missing_values = churn_df.isnull().sum()
print(missing_values)
print()

churn_distribution = churn_df['churn'].value_counts(normalize=True)
print(churn_distribution)
print()

# Encoding categorical features
encoder = OneHotEncoder(sparse=False, drop='first')
categorical_vars = churn_df.select_dtypes(include=['object', 'category']).columns.tolist()
encoded_features = encoder.fit_transform(churn_df[categorical_vars])
encoded_feature_names = encoder.get_feature_names_out(categorical_vars)

encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=churn_df.index)

churn_df_encoded = churn_df.drop(columns=categorical_vars).join(encoded_df)

# Data preprocessing
X = churn_df_encoded.drop(columns=['churn'])
y = churn_df_encoded['churn']

scaler = StandardScaler().set_output(transform='pandas')
X_scaled = scaler.fit_transform(X)

features_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=churn_df.index)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, y, test_size=0.2, random_state=42)

# Models
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)

logreg_pred_series = pd.Series(logreg_pred)
print(logreg_pred_series.value_counts())
print()

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_pred_series = pd.Series(rf_pred)
print(rf_pred_series.value_counts())

logreg_cr = classification_report(y_test, logreg_pred)
logreg_cm = confusion_matrix(y_test, logreg_pred)

rf_cr = classification_report(y_test, rf_pred)
rf_cm = confusion_matrix(y_test, rf_pred)

logreg_accuracy = logreg_cm.trace() / logreg_cm.sum()
rf_accuracy = rf_cm.trace() / rf_cm.sum()

higher_accuracy = "LogisticRegression" if logreg_accuracy > rf_accuracy else "RandomForest"
print(higher_accuracy)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Logistic Regression Confusion Matrix
sns.heatmap(logreg_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Random Forest Confusion Matrix
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Random Forest Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Plot classification reports
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Logistic Regression Classification Report
logreg_cr_dict = classification_report(y_test, logreg_pred, output_dict=True)
sns.heatmap(pd.DataFrame(logreg_cr_dict).iloc[:-1, :].T, annot=True, cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression Classification Report')

# Random Forest Classification Report
rf_cr_dict = classification_report(y_test, rf_pred, output_dict=True)
sns.heatmap(pd.DataFrame(rf_cr_dict).iloc[:-1, :].T, annot=True, cmap='Blues', ax=axes[1])
axes[1].set_title('Random Forest Classification Report')

plt.tight_layout()
plt.show()
