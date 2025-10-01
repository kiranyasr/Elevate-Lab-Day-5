"""
tree_workflow.py
Train and evaluate Decision Tree & Random Forest on heart.csv
Outputs saved in ../outputs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "heart.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- Load Data ----------
df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df = df.dropna(axis=1, how="all")

target_col = "target" if "target" in df.columns else df.columns[-1]
y = df[target_col]
X = df.drop(columns=[target_col])

# Convert non-numeric to numeric if any
for col in X.select_dtypes(include=["object", "category"]).columns:
    X[col] = pd.factorize(X[col])[0]
# Fill missing numeric values
for col in X.select_dtypes(include=[np.number]).columns:
    X[col] = X[col].fillna(X[col].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
)

# ---------- Baseline Decision Tree ----------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print("Baseline Decision Tree accuracy:", accuracy_score(y_test, dt.predict(X_test)))

# ---------- Tune max_depth ----------
best_d, best_score = None, -1
for d in range(1, 16):
    scores = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=42), X, y, cv=5)
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_d = d
print("Best max_depth (CV):", best_d, "CV Acc:", best_score)

dt_tuned = DecisionTreeClassifier(max_depth=best_d, random_state=42)
dt_tuned.fit(X_train, y_train)
print("Tuned DT accuracy:", accuracy_score(y_test, dt_tuned.predict(X_test)))

plt.figure(figsize=(14, 10))
plot_tree(dt_tuned, feature_names=X.columns, class_names=[str(c) for c in sorted(y.unique())], max_depth=4)
plt.title(f"Tuned Decision Tree (max_depth={best_d})")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "decision_tree.png")

# ---------- Random Forest ----------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest accuracy:", accuracy_score(y_test, rf.predict(X_test)))

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop features (RF):\n", importances.head(10))

plt.figure(figsize=(10, 6))
importances.head(12).plot(kind="bar")
plt.title("Feature Importances (RF)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rf_importances.png")

# ---------- Cross-validation comparison ----------
cv_dt = cross_val_score(dt_tuned, X, y, cv=5)
cv_rf = cross_val_score(rf, X, y, cv=5)
print(f"CV Acc - Tuned DT: {cv_dt.mean():.4f}, RF: {cv_rf.mean():.4f}")

# ---------- Save Models ----------
joblib.dump(dt_tuned, OUTPUT_DIR / "dt_tuned.joblib")
joblib.dump(rf, OUTPUT_DIR / "rf_model.joblib")

print("\n✅ Done! Outputs saved in:", OUTPUT_DIR)
