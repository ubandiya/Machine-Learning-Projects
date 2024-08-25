import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Customer_Churn.csv')
df = pd.DataFrame(df)

X = df.drop(columns=['Churn'])
y = df['Churn'].map({0: 'No churn', 1: 'Churn'})
print(f'X Shape: {X.shape}')
print(f'Y Shape: {y.shape}')
print()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f'X_train shape: {X_train.shape}')
print(f'y_train.shape: {y_train.shape}')
print()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)
classf = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion matrix:\n {conf}')
print(f'Classification report:\n {classf}')

sns.heatmap(conf, annot=True, fmt='d', cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

k_values = list(range(1, 31))
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=5)
    cv_scores.append(scores.mean())

print(f'The best value of k: {k_values[cv_scores.index(max(cv_scores))]}')
y_pred_cv = cross_val_predict(knn, X_scaled, y, cv=5)
cv_accuracy = accuracy_score(y, y_pred_cv)
print(f'CV Accuracy: {cv_accuracy}')