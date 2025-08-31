# predict_ultra_plan.py

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Paths & Data Load

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # go up one folder from scripts
DATA_DIR = PROJECT_ROOT / "dataset"

csv_path = DATA_DIR / "users_behavior.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"Could not find dataset at: {csv_path}")

data = pd.read_csv(csv_path).drop_duplicates().dropna()


# Features / Target

if "is_ultra" not in data.columns:
    raise KeyError("Expected target column 'is_ultra' in users_behavior.csv")

X = data.drop(columns=["is_ultra"])
y = data["is_ultra"].astype(int)
feature_names = X.columns.tolist()

# Stratified 60/20/20 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)


# Model Zoo (with proper scaling where needed)

models = {
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("clf", GradientBoostingClassifier(random_state=42))
    ]),
    # linear models / SVM need scaling
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler(with_mean=False) if hasattr(X, "sparse") else StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler(with_mean=False) if hasattr(X, "sparse") else StandardScaler()),
        ("clf", SVC(kernel="linear", class_weight="balanced", probability=True, random_state=42))
    ]),
}

param_grids = {
    "Decision Tree": {
        "clf__max_depth": [4, 6, 8, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 5],
    },
    "Random Forest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2, 3],
    },
    "Logistic Regression": {
        "clf__C": [0.5, 1.0, 2.0],
        "clf__penalty": ["l2"],
        # solver auto-picked by sklearn for l2; could set to "lbfgs"
    },
    "Support Vector Machine": {
        "clf__C": [0.5, 1.0, 2.0],
        "clf__kernel": ["linear"],  # keep linear for feature interpretability
    },
}


# Train & Validate (GridSearch on F1)

performance_rows = []
trained_models = {}
val_preds_by_model = {}

for name, pipe in models.items():
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grids[name],
        scoring="f1",            # focus on positive class performance
        cv=5,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    trained_models[name] = best

    y_val_pred = best.predict(X_val)
    val_preds_by_model[name] = y_val_pred

    rep = classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)
    performance_rows.append({
        "Model": name,
        "Val Accuracy": round(rep["accuracy"], 3),
        "Val Precision (Ultra=1)": round(rep["1"]["precision"], 3),
        "Val Recall (Ultra=1)": round(rep["1"]["recall"], 3),
        "Val F1 (Ultra=1)": round(rep["1"]["f1-score"], 3),
    })

performance_df = pd.DataFrame(performance_rows).sort_values(by="Val F1 (Ultra=1)", ascending=False)
print("\n=== Validation Performance (sorted by F1 on Ultra) ===")
print(performance_df.to_string(index=False))


# Pick Best by Validation F1

best_name = performance_df.iloc[0]["Model"]
best_model = trained_models[best_name]
print(f"\nBest model by validation F1: {best_name}")


# Final Test Evaluation

y_test_pred = best_model.predict(X_test)
print(f"\n=== Final Test Report ({best_name}) ===")
print(classification_report(y_test, y_test_pred, digits=2, zero_division=0))

cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
print("\nConfusion Matrix (Test):")
print(cm_df)


# (Optional) Plot prediction distributions

plt.figure(figsize=(7, 4))
for label, preds in val_preds_by_model.items():
    plt.hist(preds, bins=2, alpha=0.55, label=label)
plt.xticks([0, 1])
plt.xlabel("Predicted Class (Validation)")
plt.ylabel("Count")
plt.title("Prediction Distributions (Validation)")
plt.legend()
plt.tight_layout()
plt.show()


# Feature Importances (tree models)

def get_final_estimator(model):
    """Return the final estimator (handles Pipeline or bare estimator)."""
    if hasattr(model, "named_steps"):
        # pipeline case
        return model.named_steps.get("clf", model)
    return model

for name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
    if name not in trained_models:
        continue
    est = get_final_estimator(trained_models[name])
    if hasattr(est, "feature_importances_"):
        importances = pd.Series(est.feature_importances_, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        importances.head(15).plot(kind="bar")
        plt.title(f"{name} — Top Feature Importances")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()


