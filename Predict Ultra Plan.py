# predict_ultra_plan.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("datasets/users_behavior.csv")

# Feature-target split
X = data.drop('is_ultra', axis=1)
y = data['is_ultra']
feature_names = X.columns.tolist()

# Split into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define models and hyperparameters
models = {
    'Decision Tree': (DecisionTreeClassifier(), {
        'max_depth': [5],
        'min_samples_split': [2],
        'min_samples_leaf': [2]
    }),
    'Random Forest': (RandomForestClassifier(), {
        'n_estimators': [50],
        'max_depth': [10],
        'min_samples_split': [5]
    }),
    'Logistic Regression': (LogisticRegression(max_iter=1000), {
        'C': [10],
        'penalty': ['l2']
    }),
    'Gradient Boosting': (GradientBoostingClassifier(), {
        'n_estimators': [100],
        'learning_rate': [0.1],
        'max_depth': [3]
    }),
    'Support Vector Machine': (SVC(class_weight='balanced'), {
        'C': [1],
        'kernel': ['linear']
    })
}

trained_models = {}
preds_dict = {}
performance_summary = []

# Train and evaluate each model
for name, (model, params) in models.items():
    grid = GridSearchCV(model, params, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)
    val_preds = grid.predict(X_val)
    trained_models[name] = grid.best_estimator_
    preds_dict[name] = val_preds

    report = classification_report(y_val, val_preds, output_dict=True, zero_division=0)
    performance_summary.append({
        'Model': name,
        'Accuracy': round(report['accuracy'], 2),
        'Precision (Ultra)': round(report['1']['precision'], 2),
        'Recall (Ultra)': round(report['1']['recall'], 2),
        'F1-score (Ultra)': round(report['1']['f1-score'], 2)
    })

# Summary DataFrame
performance_df = pd.DataFrame(performance_summary)
print("\nModel Performance Summary:")
print(performance_df)

# Plot prediction distributions
plt.figure(figsize=(8, 5))
for label, preds in preds_dict.items():
    plt.hist(preds, label=label, alpha=0.6, bins=2)
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Prediction Distributions by Model")
plt.legend()
plt.tight_layout()
plt.show()

# Confusion matrices
for name, model in trained_models.items():
    test_preds = model.predict(X_test)
    cm = confusion_matrix(y_test, test_preds)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    print(f"\n{name} Confusion Matrix:")
    print(cm_df)

# Feature importance
for name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
    model = trained_models[name]
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        importances.plot(kind='bar', color='teal')
        plt.title(f"{name} - Feature Importance")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.show()

# Final evaluation
print("\n✅ Final Evaluation on Test Set (Random Forest):")
final_model = trained_models['Random Forest']
test_preds = final_model.predict(X_test)
print(classification_report(y_test, test_preds))


