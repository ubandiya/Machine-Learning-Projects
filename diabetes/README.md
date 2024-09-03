# Diabetes Prediction Model Project

## Overview

This project aims to predict the likelihood of diabetes in patients using various machine learning models. The workflow includes data preprocessing, model training, hyperparameter tuning, and evaluation. The objective is to select the best-performing model based on accuracy and other performance metrics and prepare it for deployment.

## Project Structure

- `diabetes.py`: The main script for data processing, model training, evaluation, and plotting.
- `diabetes.csv`: The dataset used for model training and evaluation.

## Setup Instructions

### Prerequisites

Ensure you have the following Python packages installed:

- numpy
- pandas
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- scipy

You can install these packages using pip:

   ``` bash
   pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn scipy
```

### Running the Project

1. **Prepare the Dataset**

   Place the `diabetes.csv` file in the same directory as `diabetes.py`.

2. **Execute the Script**

   Run the `diabetes.py` script to perform the following operations:

   - Load and preprocess the data.
   - Split the data into training and testing sets.
   - Scale the features.
   - Optimize hyperparameters for various models using SMOTE.
   - Evaluate models and print performance metrics.
   - Plot model comparison results and ROC/Precision-Recall curves.

   To run the script, use:

   ``` bash
    python diabetes.py
   ```

## Usage

### Model Evaluation

The script evaluates several models, including:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

For each model, the script performs hyperparameter tuning with SMOTE to handle class imbalance and evaluates the model using accuracy, confusion matrix, and classification report.

### Plotting

The script generates plots for:

- Model comparison based on accuracy scores.
- ROC curves and Precision-Recall curves for each model.

These plots help visualize model performance and compare different models.

## Additional Notes

- **Best Model**: Based on the evaluation, the Random Forest model performed the best with the highest accuracy score.
- **Threshold Adjustment**: The script calculates the optimal threshold for Logistic Regression based on the F1 score.

## Future Work

- **Model Interpretability**: Explore tools and techniques for interpreting model predictions.
- **Deployment**: Implement the best model into a web application or API for real-time predictions.
- **Advanced Techniques**: Investigate advanced models and techniques for further improvement.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
