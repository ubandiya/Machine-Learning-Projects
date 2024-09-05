# AUTHOR: UBANDIYA Najib Yusuf

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginicia'})

print(df.head())
print()

print(df.info())
print()

print(df.describe())
print()

print(df.drop(columns='species', axis=1).skew())
print()

plt.figure(figsize=(20, 10))
sns.pairplot(df, hue='species', diag_kind='hist', height=2.5)
plt.show()

plt.figure(figsize=(20, 10))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', data=df)
plt.title('sepal width (cm) vs sepal length (cm)')
plt.show()

plt.figure(figsize=(15, 10))
sns.heatmap(df.drop(columns='species').corr(), annot=True, cmap='coolwarm')
plt.show()

X = df.drop(columns='species', axis=1)
y = df['species']

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Training set shape: {X_train.shape}')
print(f'Testing set shape: {X_test.shape}')

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3))
])

pipeline.fit(X_train, y_train)

y_pred_knn = pipeline.predict(X_test)
print(f'{'='*5} Performance {'='*5}')
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))
print(f'Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}\n')
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print()


k_values = list(range(1, 31))
cv_scores = []

for k in k_values:
    knn_ = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_values[cv_scores.index(max(cv_scores))]
print(f'Best k value: {best_k}')

sample_flower = pd.DataFrame({
    'sepal length (cm)': [5.0],
    'sepal width (cm)': [3.5],
    'petal length (cm)': [2.2],
    'petal width (cm)': [0.2]
})
predicted_species = pipeline.predict(sample_flower)
print(f'Predicted species: {predicted_species}')
print()

param_grid = {
    'knn__n_neighbors': [3, 5, 7],
    'knn__weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X, y)

print(f'Best Parameters: {grid_search.best_params_}\n'
      f'Best Score: {grid_search.best_score_}')

best_knn = pipeline.set_params(**grid_search.best_params_)
best_knn.fit(X_train, y_train)

y_pred_final = best_knn.predict(X_test)

print(f'{"="*5} Final Performance {"="*5}')
print(f'Accuracy: {accuracy_score(y_test, y_pred_final)}')
print(classification_report(y_test, y_pred_final))
print(f'Confusion Matrix:\n {confusion_matrix(y_test, y_pred_final)}')

plt.figure(figsize=(15, 10))
sns.heatmap(confusion_matrix(y_test, y_pred_final), annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
