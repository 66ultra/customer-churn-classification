# Customer Churn Prediction at SyriaTel

## Project Overview

This project aims to build a machine learning model to predict customer churn for SyriaTel, a telecommunications company. By identifying at-risk customers, the Customer Retention Team can take actions such as offering discounts or promotions to retain them.

## Dataset

The dataset used for this project is from [Kaggle: Churn in Telecoms Dataset](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset), containing 3333 customer records and 19 features related to customer behavior and usage.

## Stakeholders

The primary stakeholders are the **Customer Retention Team** at SyriaTel, focusing on reducing churn and maintaining revenue.

## Objective

The objective of this project is to create a machine learning classifier to accurately predict whether a customer will churn, thus allowing the business to intervene proactively.

## Methods and Models

The following steps were taken to complete the project:

- Data cleaning and preprocessing, including handling missing values, feature scaling, and encoding categorical variables.
- Exploratory Data Analysis (EDA) to understand key patterns and correlations.
- Baseline model with Logistic Regression.
- Class balancing techniques like class-weight adjustments and SMOTE.
- Hyperparameter tuning of a Decision Tree model, including max depth, min_samples_split, min_samples_leaf, and max_features.
- Model evaluation based on accuracy, recall, and precision to optimize for identifying churners.

## Results

The final model achieved a **94% accuracy** and **69% recall** for the churn class. This balance allows the team to catch more at-risk customers while maintaining overall model performance.

## Business Recommendations

Based on the model, the **Customer Retention Team** can:

- Focus retention efforts on customers predicted to churn, particularly those with frequent customer service interactions and high international plan usage.
- Tailor promotions and discounts for at-risk customers to reduce churn rate.