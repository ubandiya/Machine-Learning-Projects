import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginicia'})

# Data preparation
X = df.drop(columns=['species'])
y = df['species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#print(pd.DataFrame(X_scaled, columns=X.columns).head())

# Train-Test-Split (80% training sets, 20% test sets)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f'Training set shape: {X_train.shape}')
print(f'Testing set shape: {X_test.shape}')
print()

# Model selection (k-Nearest Neighbor)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Model evaluation
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Classification report:\n{classification_report(y_test, y_pred)}')
print(f'Confusion matrix:\n {confusion_matrix(y_test, y_pred)}')
print()

# Make prediction
sample_flower = pd.DataFrame({
    'sepal length (cm)': [5.0],
    'sepal width (cm)': [3.5],
    'petal length (cm)': [2.2],
    'petal width (cm)': [0.2]
})
sample_flower_scaled = scaler.transform(sample_flower)
predicted_species = knn.predict(sample_flower_scaled)
print(f'Predicted species: {predicted_species}')
print()

# Cross-validation for different values of k
k_values = list(range(1, 31))
cv_scores = []

for k in k_values:
    knn_ = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_values[cv_scores.index(max(cv_scores))]
print(f'Best k value: {best_k}')

#sns.scatterplot(x=k_values, y=cv_scores)
sns.scatterplot(x=k_values, y=cv_scores, hue=cv_scores)
plt.xlabel('k')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN Cross-Validation Accuracy vs k')
plt.show()
