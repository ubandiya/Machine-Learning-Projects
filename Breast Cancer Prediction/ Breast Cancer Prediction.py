# AUTHOR: UBANDIYA Najib Yusuf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())
print()

print(df.info())
print()

print(df.describe())
print()

plt.figure(figsize=(20, 10))
sns.boxplot(data=df.drop('target', axis=1))
plt.xticks(rotation=90)
plt.title('Box Plot of Features')
plt.show()

z_scores = stats.zscore(df.drop('target', axis=1))
abs_z_scores = np.abs(z_scores)
outliers = (abs_z_scores > 3).any(axis=1)
df_outliers = df[outliers]
df = df[~outliers]

features = df.drop(columns='target')
targets = df['target']

selector = SelectKBest(f_classif, k=10)
X_new = selector.fit(features, targets)
print(f'Selected features: {features.columns[selector.get_support()]}')

X = df[['mean radius', 'mean perimeter', 'mean area', 'mean concavity',
        'mean concave points', 'worst radius', 'worst perimeter', 'worst area',
       'worst concavity', 'worst concave points']]
y = df['target'].map({0: 'No cancer', 1: 'Cancer'})

df_visual = X.copy()
df_visual['target'] = y

sns.pairplot(df_visual, hue='target', diag_kind='kde')
plt.show()

plt.figure(figsize=(16, 12))
for i, column in enumerate(X.columns, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(x='target', y=column, data=df_visual)
    plt.title(f'Boxplot of {column}')
    plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Selected Features')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3))
])

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier())
])

knn_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

y_pred_knn = knn_pipeline.predict(X_test)
print(f'{'='*5} KNN Performance {'='*5}')
print(f'Accuracy: {accuracy_score(y_test, y_pred_knn)}')
print(classification_report(y_test, y_pred_knn))
print(f'Confusion Matrix:\n {confusion_matrix(y_test, y_pred_knn)}')
knn_cv_scores = cross_val_score(knn_pipeline, X, y, cv=5)
print(f'KNN Cross-Validation Scores: {knn_cv_scores}')
print()

y_pred_rf = rf_pipeline.predict(X_test)
print(f'{'='*5} Random Forest Performance {'='*5}')
print(f'Accuracy: {accuracy_score(y_test, y_pred_rf)}')
print(classification_report(y_test, y_pred_rf))
print(f'Confusion Matrix:\n {confusion_matrix(y_test, y_pred_rf)}')
rf_cv_scores = cross_val_score(rf_pipeline, X, y, cv=5)
print(f'Random Forest Cross-Validation Scores: {rf_cv_scores}')
print()

param_grid_knn = {
    'knn__n_neighbors': [3, 5, 7],
    'knn__weights': ['uniform', 'distance']
}

param_grid_rf = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_leaf': [2,3, 5]
}

knn_grid_search = GridSearchCV(knn_pipeline, param_grid_knn, cv=5)
rf_grid_search = GridSearchCV(rf_pipeline, param_grid_rf, cv=5)

knn_grid_search.fit(X, y)
rf_grid_search.fit(X, y)

print(f'Best KNN Parameters: {knn_grid_search.best_params_}\n'
      f'Best Score: {knn_grid_search.best_score_}')
print()
print(f'Best RF Parameters: {rf_grid_search.best_params_}\n'
      f'Best Score: {rf_grid_search.best_score_}')
print()

feature_importances = rf_pipeline.named_steps['rf'].feature_importances_
features_ = X.columns
importance_df = pd.DataFrame({
    'Features': features_,
    'Importances': feature_importances
})
importance_df = importance_df.sort_values(by='Importances', ascending=False)
print(importance_df)

# Re-fit with best parameters for KNN and Random Forest
best_knn = knn_pipeline.set_params(**knn_grid_search.best_params_)
best_rf = rf_pipeline.set_params(**rf_grid_search.best_params_)

best_knn.fit(X_train, y_train)
best_rf.fit(X_train, y_train)

y_pred_knn_final = best_knn.predict(X_test)
y_pred_rf_final = best_rf.predict(X_test)

print(f'{"="*5} Final KNN Performance {"="*5}')
print(f'Accuracy: {accuracy_score(y_test, y_pred_knn_final)}')
print(classification_report(y_test, y_pred_knn_final))
print(f'Confusion Matrix:\n {confusion_matrix(y_test, y_pred_knn_final)}')
print()

print(f'{"="*5} Final Random Forest Performance {"="*5}')
print(f'Accuracy: {accuracy_score(y_test, y_pred_rf_final)}')
print(classification_report(y_test, y_pred_rf_final))
print(f'Confusion Matrix:\n {confusion_matrix(y_test, y_pred_rf_final)}')

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No cancer', 'Cancer'],
                yticklabels=['No cancer', 'Cancer'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Confusion matrices
cm_knn = confusion_matrix(y_test, y_pred_knn, labels=['No cancer', 'Cancer'])
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=['No cancer', 'Cancer'])

# Plot confusion matrices
plot_confusion_matrix(cm_knn, 'KNN Confusion Matrix')
plot_confusion_matrix(cm_rf, 'Random Forest Confusion Matrix')
