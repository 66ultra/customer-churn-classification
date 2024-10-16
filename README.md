# Customer Churn Prediction Project

## Project Overview

This project aims to predict customer churn for a telecommunications company using machine learning models. The goal is to identify customers who are likely to churn and provide insights that can help the business take proactive measures.

The final model was built using a Decision Tree Classifier, with hyperparameter tuning applied to optimize performance. The project includes feature engineering, data preprocessing, model building, and evaluation, along with visualizations that provide actionable insights.

## Dataset

- **Source**: The dataset was sourced from Kaggle.
- **Columns**: The dataset consists of various customer details such as:
  - Total day charge
  - Customer service calls
  - International plan
  - Voice mail plan
  - Many other telecommunication features

The target variable is **churn**, which indicates whether a customer has churned (True) or not (False).

## Key Steps

1. **Data Preprocessing**:
   - Removed unnecessary features such as `state` and `phone_number`.
   - Scaled the numeric features for better model performance.
   - Addressed class imbalance using SMOTE.

2. **Model Building and Evaluation**:
   - Tried Logistic Regression, Decision Tree, and tuned the Decision Tree for better performance.
   - Used metrics such as Accuracy, Precision, Recall, and F1-Score to evaluate model performance.
   - Applied hyperparameter tuning to find the optimal parameters for the Decision Tree.

3. **Visualizations**: Created visualizations to highlight feature importance, model comparisons, and the final results.

## Key Results

- **Final Model**: The optimal model selected was the Decision Tree Classifier with hyperparameters:
  - `max_depth=12`
  - `min_samples_split=5`
  - `min_samples_leaf=5`
  - `max_features=14`
  
- **Model Performance**:
  - Accuracy: 94%
  - Precision for Churn Class: 83%
  - Recall for Churn Class: 72%
  - F1-Score for Churn Class: 77%

## Visualizations

Here are some key visualizations created during the project:

### 1. Feature Importance

This bar chart displays the most important features contributing to customer churn predictions in the Decision Tree model.

![Feature Importance](../customer-churn-classification/figures/feature_importance_optimal_decision_tree.png)

### 2. Confusion Matrix (Heatmap)

This heatmap shows the confusion matrix of the optimal Decision Tree model, which provides insights into true positives, false positives, true negatives, and false negatives.

![Confusion Matrix](../customer-churn-classification/figures/confusion_matrix_optimal_decision_tree.png)

### 3. Model Comparison

A bar chart comparing Logistic Regression and the optimized Decision Tree models in terms of Precision, Recall, and F1-Score.

![Model Comparison](../customer-churn-classification/figures/comparison_logreg_dtree.png)

## Saved Models

To use the saved models, you can load them from the `models` folder.
