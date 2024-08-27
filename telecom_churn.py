# AUTHOR: UBANDIYA Najib Yusuf

import pandas as pd
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

encoder = OneHotEncoder(sparse=False, drop='first')
categorical_vars = churn_df.select_dtypes(include=['object', 'category']).columns.tolist()
encoded_features = encoder.fit_transform(churn_df[categorical_vars])
encoded_feature_names = encoder.get_feature_names_out(categorical_vars)

encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=churn_df.index)

churn_df_encoded = churn_df.drop(columns=categorical_vars).join(encoded_df)

X = churn_df_encoded.drop(columns=['churn'])
y = churn_df_encoded['churn']

scaler = StandardScaler().set_output(transform='pandas')
X_scaled = scaler.fit_transform(X)

features_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=churn_df.index)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, y, test_size=0.2, random_state=42)

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
