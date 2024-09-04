# AUTHOR: UBANDIYA Najib Yusuf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif

data = load_breast_cancer()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

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

X = df[['mean radius', 'mean perimeter', 'mean area', 'mean concavity',
       'mean concave points', 'area error', 'worst radius', 'worst perimeter',
       'worst area', 'worst concave points']]
y = df['target'].map({0: 'No cancer', 1: 'Cancer'})

selector = SelectKBest(f_classif, k=10)
X_new = selector.fit(X, y)
print(f'Selected features: {X.columns[selector.get_support()]}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test_scaled)

knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
class_ = classification_report(y_test, y_pred)
conf_ = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification report:\n{class_}')
print(f'Confusion matrix:\n {conf_}')
print(y_pred)

k_values = list(range(1, 31))
cv_scores = []

for k in k_values:
    knn_ = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

print(f'The best value for k: {k_values[cv_scores.index(max(cv_scores))]}')

sns.heatmap(conf_, annot=True, fmt='d', cmap='Blues', xticklabels=['No cancer', 'Cancer'], yticklabels=['No cancer', 'Cancer'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Cofusion Matrix')
plt.show()

synth_data = np.random.normal(loc=np.mean(_test_scaled, axis=0), scale=np.std(X_test_scaled, axis=0), size=(30, X_test_scaled.shape[1]))
synth_scaled = scaler.transform(synth_data)
synth_pred = knn.predict(synth_scaled)
print(f'Accuracy of synthetic data prediction: {accuracy_score(y_test, synth_pred)}')
