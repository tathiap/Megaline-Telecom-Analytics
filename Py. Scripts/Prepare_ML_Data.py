#!/usr/bin/env python
# coding: utf-8

# ## Prepare Machine Learning Dataset

# The purpose of this analysis is to evaluate the performance of various machine learning models (Decision Tree, Random Forest, Logistic Regression) in predicting Ultra plan subscriptions. By examining metrics like accuracy, precision, recall, and feature importance, we aim to understand which features most influence user behavior and which model performs best.

# In[1]:


get_ipython().system('python load_megaline_data.ipynb')


# In[2]:


import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score


# In[3]:


# Connect to MySQL and load data
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="September97!",  # Your MySQL password
    database="megaline_db"
)

# Load the combined features view
df = pd.read_sql("SELECT * FROM megaline_features;", conn)
conn.close()

print("Data Preview:")
display(df.head())


# In[4]:


# Prepare ML Data

# Review columns
print("Available columns in the dataset:")
print(df.columns.tolist())

# Remove any columns that directly leak the target
# (e.g., 'is_ultra', and revenue columns if they directly encode the plan price)
leakage_columns = ['is_ultra', 'user_id', 'total_revenue', 'avg_revenue']
feature_columns = [col for col in df.columns if col not in leakage_columns]

# Create features (X) and target (y)

X = df[feature_columns]
y = df['is_ultra']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")


# In[5]:


# Train Baseline Models
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

trained_models = {}
performance_summary = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    trained_models[name] = model
    
    # Evaluate model
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    performance_summary.append({
        "Model": name,
        "Accuracy": round(report['accuracy'], 2),
        "Precision (Ultra)": round(report['1']['precision'], 2),
        "Recall (Ultra)": round(report['1']['recall'], 2),
        "F1-score (Ultra)": round(report['1']['f1-score'], 2)
    })

performance_df = pd.DataFrame(performance_summary)
display(performance_df)


# ### Model Evaluation
# #### Feature Importance, Confusion Matrix, ROC Curve

# In[6]:


# Feature Importance for Random Forest
rf_model = trained_models['Random Forest']
importances = pd.Series(rf_model.feature_importances_, index=X.columns)

top_features = importances.sort_values(ascending=False).head(5)
print("Top 5 Features Influencing Ultra Plan Prediction:")
display(top_features)

plt.figure(figsize=(8,5))
importances.sort_values(ascending=True).plot(kind='barh', title="Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# In[7]:


# Confusion Matrices for All Models
for name, model in trained_models.items():
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Ultra", "Ultra"], yticklabels=["Not Ultra", "Ultra"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# In[8]:


# ROC Curve for Random Forest (Best Model)
rf_model = trained_models['Random Forest']
rf_probs = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, rf_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='blue')
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest (No Leakage)')
plt.legend()
plt.grid(True)
plt.show()


# ### Key Insights Summary
# 
# Model Performance:
# * Random Forest achieved the best performance with an accuracy of 0.86 and the highest F1-score (0.73) for predicting Ultra plan subscriptions.
# 
# * Decision Tree performed moderately well (accuracy 0.81) but was less robust compared to Random Forest.
# 
# * Logistic Regression had the lowest recall (0.23) and struggled with class imbalance, though its precision for Ultra was relatively high (0.85).
# 
# Feature Importance: 
# 
# * The top predictors for Ultra plan subscriptions were mb_used, avg_mb, and avg_minutes, indicating that mobile data and call duration patterns strongly influence subscription behavior.
# 
# * Features like calls and messages contributed less to the model.
# 
# ROC Curve:
# 
# * The ROC curve for Random Forest yielded an AUC of 0.88, reflecting strong performance in distinguishing between Ultra and non-Ultra subscribers.
# 
# Confusion Matrix:
# 
# * Random Forest showed fewer false positives and false negatives compared to other models, confirming its superior classification ability.
# 

# In[ ]:




