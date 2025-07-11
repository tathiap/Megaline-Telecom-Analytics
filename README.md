# Predicting User Subscription Behavior — Megaline Plan Analysis & Machine Learning Classification

## Overview: 

This repository contains two interconnected data analysis and machine learning projects based on subscriber behavior data from a fictional telecom company, Megaline. The first project focuses on exploratory data analysis (EDA), A/B testing, and revenue forecasting. The second project applies machine learning classification models to predict whether a user is likely to subscribe to the premium "Ultra" plan.

### Project 1: Megaline Revenue Plan Analysis


#### Objective:
Analyze customer behavior and revenue across Surf and Ultimate prepaid plans to identify which plan generates more revenue and what regions or usage patterns contribute to these outcomes.

#### Techniques Used:
* Data Cleaning & Preprocessing
* Explora tory Data Analysis (EDA)
* Revenue Breakdown & Visualizations
* Hypothesis Testing (T-tests)
* Revenue Forecasting (Linear Trend Analysis)

#### Key Findings:
* Ultimate plan users tend to generate higher revenue per user but are fewer in number.
* Significant revenue differences were observed between plans and across regions.
* Internet usage and call minutes showed different usage behaviors by plan.
* Forecasting indicated potential seasonal or monthly trends in usage metrics.

### Project 2: Predicting Ultra Plan Subscriptions (ML Classification)

#### Objective:
Build and evaluate machine learning models to predict whether a user will subscribe to the Ultra plan based on their monthly usage patterns (calls, texts, and data usage).

#### Models Implemented:
* Decision Tree
* Random Forest
* Logistic Regression
* Gradient Boosting
* Support Vector Machine

### Workflow:
* Data Cleaning & Splitting (Train/Val/Test)
* Hyperparameter Tuning with GridSearchCV
* Evaluation with Classification Report, Confusion Matrix, and Accuracy Score
* Feature Importance Visualization
* Final Model Selection and Performance Summary

### Final Model:
* Random Forest delivered the best overall performance with 81% test accuracy and strong precision for Ultra user prediction.
* Tree-based models offered interpretable insights via feature importance plots.

### Dataset
The dataset users_behavior.csv includes:

* Minutes, messages, and MB of mobile data used per user
* Binary target variable: is_ultra (1 = Ultra plan subscriber, 0 = Surf)


