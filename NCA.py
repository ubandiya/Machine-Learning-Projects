# Import required libraries
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Function to plot 2D projections
def plot_2d_projection(X_proj, y, title):
    plt.figure()
    colors = ['r', 'g', 'b']

    for i, color in zip(range(len(iris.target_names)), colors):
        idx = np.where(y == i)[0]
        sns.scatterplot(x=X_proj[idx, 0], y=X_proj[idx, 1], color=color,
                        label=iris.target_names[i], edgecolor='k', s=50)

    plt.title(title)
    plt.legend()
    plt.show()

# Apply PCA and reduce dimensionality to 2
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Apply NCA and reduce dimensionality to 2
nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
nca.fit(X_train, y_train)
X_train_nca = nca.transform(X_train)
X_test_nca = nca.transform(X_test)

# Initialize k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train and test on PCA-transformed data
knn.fit(X_train_pca, y_train)
y_pred_pca = knn.predict(X_test_pca)
pca_accuracy = accuracy_score(y_test, y_pred_pca)

# Train and test on NCA-transformed data
knn.fit(X_train_nca, y_train)
y_pred_nca = knn.predict(X_test_nca)
nca_accuracy = accuracy_score(y_test, y_pred_nca)

# Print accuracy results
print(f"Accuracy with PCA: {pca_accuracy * 100:.2f}%")
print(f"Accuracy with NCA: {nca_accuracy * 100:.2f}%")
print()

print(f'PCA confusion matrix:\n{confusion_matrix(y_test, y_pred_pca)}')
print()
print(f'NCA confusion matrix:\n {confusion_matrix(y_test, y_pred_nca)}')
print()

print(f'PCA classification report:\n {classification_report(y_test, y_pred_pca)}')
print()
print(f'NCA classification report:\n {classification_report(y_test, y_pred_nca)}')

# Plot the projections
#plot_2d_projection(X_train_pca, y_train, 'PCA: 2D Projection of Iris Dataset')
#plot_2d_projection(X_train_nca, y_train, 'NCA: 2D Projection of Iris Dataset')
