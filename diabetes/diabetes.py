import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_curve, roc_auc_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RF, GradientBoostingClassifier as GB
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    column = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Replace 0 values with median in specified columns
    mask = (df[column] == 0).any(axis=1)
    df[column] = df[column].replace(0, df[column].median())
    
    # Apply log transformation
    df['DiabetesPedigreeFunction'] = np.log1p(df['DiabetesPedigreeFunction'])
    df['Insulin'] = np.log1p(df['Insulin'])
    
    # Handle NaN and infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    
    return df

def clean_params(model_name, best_params):
    # Strip the 'model__' prefix from parameter names
    return {key.replace('model__', ''): value for key, value in best_params[model_name].items()}

def define_models():
    models = {
        'KNN': KNN(),
        'Logistic Regression': LR(max_iter=1000),
        'Decision Tree': DTC()
    }
    return models

def define_ensemble_models():
    models = {
        'Random Forest': RF(n_estimators=100, random_state=42),
        'Gradient Boosting': GB(n_estimators=100, random_state=42)
    }
    return models

def get_all_models():
    models = define_models()
    ensemble_models = define_ensemble_models()
    models.update(ensemble_models)
    return models

def tune_hyperparameters(models, X_train, y_train):
    param_grids = {
        'KNN': {'model__n_neighbors': randint(1, 21), 'model__weights': ['uniform', 'distance'], 'model__p': [1, 2]},
        'Logistic Regression': {'model__C': uniform(0.001, 10), 'model__solver': ['liblinear', 'lbfgs'], 'model__penalty': ['l1', 'l2']},
        'Decision Tree': {'model__criterion': ['gini', 'entropy'], 'model__max_depth': randint(1, 21), 'model__min_samples_split': randint(2, 21), 'model__min_samples_leaf': randint(1, 21)},
        'Random Forest': {'model__n_estimators': randint(50, 200), 'model__max_depth': randint(5, 20), 'model__min_samples_split': randint(2, 10), 'model__min_samples_leaf': randint(1, 5)},
        'Gradient Boosting': {'model__n_estimators': randint(50, 200), 'model__learning_rate': uniform(0.01, 0.3), 'model__max_depth': randint(3, 10)}
    }
    
    best_params = {}
    for model_name, model in models.items():
        print(f'Optimizing {model_name} with SMOTE...')
        pipeline = ImbPipeline([('smote', SMOTE()), ('model', model)])
        random_search = RandomizedSearchCV(pipeline, param_distributions=param_grids[model_name], n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1, verbose=1)
        random_search.fit(X_train, y_train)
        best_params[model_name] = random_search.best_params_
        print(f'Best parameters for {model_name}: {best_params[model_name]}')
        print(f'Best score for {model_name}: {random_search.best_score_}')
        print()
    return best_params

def evaluate_models(best_params, X_test, y_test, X_train, y_train):
    all_models = get_all_models()  # Get all models including ensemble models
    for model_name, params in best_params.items():
        print(f'Evaluating {model_name}...')
        pipeline = ImbPipeline([
            ('smote', SMOTE()),
            ('model', all_models[model_name])  # Use all_models instead of define_models()
        ])
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f'Confusion Matrix for {model_name}:')
        print(confusion_matrix(y_test, y_pred))
        print(f'Accuracy Score for {model_name}: {accuracy_score(y_test, y_pred)}')
        print(f'Classification Report for {model_name}:')
        print(classification_report(y_test, y_pred))
        print()

def plot_results(best_params, X_train, y_train, X_test, y_test):
    all_models = get_all_models()  # Get all models including ensemble models
    scores = []
    
    for model_name, params in best_params.items():
        pipeline = ImbPipeline([
            ('smote', SMOTE()),
            ('model', all_models[model_name])  # Use all_models instead of define_models()
        ])
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    
    sns.barplot(x=list(best_params.keys()), y=scores)  # Use sns.barplot instead of plt.bar
    plt.xlabel('Model')
    plt.ylabel('Accuracy Score')
    plt.title('Model Comparison')
    plt.show()

def adjust_threshold(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    # Prevent division by zero by adding a small constant to the denominator
    epsilon = 1e-10
    f1_scores = 2 * (precision * recall) / (precision + recall + epsilon)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f'Best threshold based on F1 score: {best_threshold}')
    return best_threshold

def plot_roc_pr_curves(model, X_test, y_test, model_name):
    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(14, 6))
    
    # Plot ROC Curve
    plt.subplot(1, 2, 1)
    sns.lineplot(x=fpr, y=tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    
    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    
    # Plot Precision-Recall Curve
    plt.subplot(1, 2, 2)
    sns.lineplot(x=recall, y=precision, label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='lower left')
    
    plt.tight_layout()
    plt.show()

# Main Workflow
df = load_and_preprocess_data('diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = get_all_models()
best_params = tune_hyperparameters(models, X_train_scaled, y_train)
evaluate_models(best_params, X_test_scaled, y_test, X_train_scaled, y_train)
plot_results(best_params, X_train_scaled, y_train, X_test_scaled, y_test)

# Adjust threshold for Logistic Regression as an example
pipeline_lr = ImbPipeline([('smote', SMOTE()), ('model', LR(max_iter=1000))])
pipeline_lr.set_params(**best_params['Logistic Regression'])
pipeline_lr.fit(X_train_scaled, y_train)
lr_best_threshold = adjust_threshold(pipeline_lr, X_test_scaled, y_test)

# Plot ROC and Precision-Recall curves for all models
models = {
    'KNN': KNN(**clean_params('KNN', best_params)),
    'Logistic Regression': LR(**clean_params('Logistic Regression', best_params)),
    'Decision Tree': DTC(**clean_params('Decision Tree', best_params)),
    'Random Forest': RF(**clean_params('Random Forest', best_params)),
    'Gradient Boosting': GB(**clean_params('Gradient Boosting', best_params))
}

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    plot_roc_pr_curves(model, X_test_scaled, y_test, model_name)
