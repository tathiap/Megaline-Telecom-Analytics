# Predicting User Subscription Behavior — Megaline Analysis & Machine Learning

## Overview
This repository contains multiple data analysis and machine learning projects built on subscriber behavior data from a fictional telecom company, **Megaline**.  

The projects aim to:
1. Analyze customer behavior and revenue trends across different plans.
2. Perform **A/B testing** to compare the Surf vs. Ultra plans.
3. Evaluate **churn and retention metrics** for user lifetime value (LTV).
4. Build **machine learning models** to predict Ultra plan subscriptions.
5. Use **SQL queries and pipelines** for efficient data handling and analysis.

---

## Project 1: Megaline Revenue & A/B Testing
**Objective:**  
Analyze customer behavior and revenue across Surf and Ultra plans to determine which plan drives higher **conversion rate (CR)** and **average revenue per user (ARPU)**.  

**Techniques Used:**  
- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- SQL Integration for A/B Metrics (CR, ARPU)  
- Hypothesis Testing (T-tests & Chi-Square Tests)  
- Visualizations (ARPU & Conversion Rate Comparisons)  

**Key Findings:**  
- Ultra plan users show a **higher ARPU** and **conversion rate** compared to Surf users.  
- Statistical testing confirmed significant differences in revenue per user (p < 0.05).

---

## Project 2: Churn & Retention Analysis
**Objective:**  
Measure **churn rate** and **retention rate** to evaluate customer loyalty and estimate user lifetime value (LTV).

**Techniques Used:**  
- SQL queries to calculate churn/retention by plan.  
- Visualization of churn vs. retention for Surf vs. Ultra.  
- Insights on user engagement and plan stickiness.

**Key Findings:**  
- Ultra plan users have **lower churn rates**, indicating better long-term retention.  
- Surf users are less engaged and more likely to churn without incentives.

---

## Project 3: Machine Learning — Predicting Ultra Plan Subscriptions
**Objective:**  
Build and evaluate machine learning models to predict whether a user will subscribe to the **Ultra plan** based on their monthly usage patterns (calls, messages, and data usage).

**Models Implemented:**  
- Decision Tree  
- Random Forest (Best Performance)  
- Logistic Regression  
- Gradient Boosting  
- Support Vector Machine (SVM)

**Workflow:**  
- Data Cleaning & Preprocessing  
- Train/Validation/Test Split  
- Hyperparameter Tuning (GridSearchCV)  
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score  
- Feature Importance Visualization  

**Final Model:**  
- **Random Forest** delivered the best performance with **81% test accuracy** and strong precision for Ultra prediction.

---

## Project 4: SQL Analysis & Data Pipelines
**Objective:**  
Leverage SQL queries and views for data exploration, revenue breakdowns, and feature generation.

**Included SQL Files:**  
- `megaline_setup.sql`: Database setup and schema creation.  
- `megaline_analysis_views.sql`: Views for A/B testing metrics.  
- `query_for_churn_rate.sql`: Churn and retention SQL logic.

---

## Dataset
**`users_behavior.csv`** includes:  
- Minutes, messages, and MB of mobile data used per user.  
- Binary target variable: `is_ultra` (1 = Ultra plan subscriber, 0 = Surf plan).

---

## How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Predicting-User-Subscription-Behavior.git
   cd Predicting-User-Subscription-Behavior
