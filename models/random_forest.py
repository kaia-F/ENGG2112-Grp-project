"""
ENGG2112 - Flood Probability (Dataset 2): Random Forest Model
=============================================================
Dataset : flood.csv (Kaggle Playground Series S4E5)
Target  : FloodProbability (continuous) binned into four classes
          using data-driven quartile cuts:
            Low        : bottom 25%
            Moderate   : 25th–50th percentile
            High       : 50th–75th percentile
            Very High  : top 25%

Set INCLUDE_DRAINAGE = True  to train with TopographyDrainage included.
Set INCLUDE_DRAINAGE = False to train without TopographyDrainage.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
)

# =========================================================
# Toggle this to include or exclude the drainage feature
# =========================================================
INCLUDE_DRAINAGE = True

# =========================================================
# 1. Load dataset — first 4000 rows only
# =========================================================
FILE_PATH = "flood.csv"

df = pd.read_csv(FILE_PATH, nrows=4000)
df.columns = df.columns.str.strip()

print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
print(f"FloodProbability range: {df['FloodProbability'].min():.3f} – {df['FloodProbability'].max():.3f}\n")

# =========================================================
# 2. Bin FloodProbability into four classes using quartiles.
#    Quartile-based binning is used because the values in this
#    dataset are tightly clustered around 0.35–0.68, so fixed
#    thresholds (0.25/0.50/0.75) would leave classes empty.
#    Quartile cuts guarantee a balanced class distribution.
# =========================================================
df["FloodClass"] = pd.qcut(
    df["FloodProbability"],
    q=4,
    labels=["Low", "Moderate", "High", "Very High"]
)

print("Class distribution after quartile binning:")
print(df["FloodClass"].value_counts().sort_index(), "\n")

# =========================================================
# 3. Encode target
# =========================================================
label_encoder = LabelEncoder()
df["FloodClass"] = label_encoder.fit_transform(df["FloodClass"])
CLASSES = label_encoder.classes_
y = df["FloodClass"]

# =========================================================
# 4. Select features based on boolean
#    TopographyDrainage is the drainage equivalent in this dataset
# =========================================================
drop_cols = ["FloodProbability", "FloodClass"]
if not INCLUDE_DRAINAGE:
    drop_cols.append("TopographyDrainage")

X = df.drop(columns=drop_cols)
feature_names = list(X.columns)

case_name = "With Drainage" if INCLUDE_DRAINAGE else "Without Drainage"

print(f"Running Random Forest — {case_name}")
print(f"Features ({len(feature_names)}): {feature_names}\n")

# =========================================================
# 5. Train / test split
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# 6. Pipeline and hyperparameter tuning
# =========================================================
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model",   RandomForestClassifier(
        random_state=42, n_jobs=-1, class_weight="balanced"
    )),
])

# n_estimators: more trees reduce variance but increase compute time
# max_depth:    None = fully grown; 10 = limits overfitting on smaller datasets
# min_samples_split: higher values prevent overfitting on noisy leaf nodes
param_grid = {
    "model__n_estimators":      [100, 200],
    "model__max_depth":         [None, 10],
    "model__min_samples_split": [2, 5],
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("[Tuning] Running GridSearchCV (10-fold CV)...")
gs = GridSearchCV(pipeline, param_grid, cv=cv,
                  scoring="accuracy", n_jobs=-1, refit=True)
gs.fit(X_train, y_train)
best_model = gs.best_estimator_

print(f"Best hyperparameters: {gs.best_params_}")

# =========================================================
# 7. Evaluate
# =========================================================
cv_scores = cross_val_score(best_model, X_train, y_train,
                            cv=cv, scoring="accuracy", n_jobs=-1)
y_pred    = best_model.predict(X_test)
test_acc  = accuracy_score(y_test, y_pred)
macro_auc = roc_auc_score(
    label_binarize(y_test, classes=list(range(len(CLASSES)))),
    best_model.predict_proba(X_test),
    multi_class="ovr", average="macro"
)

print(f"\nTest Accuracy        : {test_acc:.4f}")
print(f"CV Accuracy (10-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Macro ROC-AUC        : {macro_auc:.4f}")
print(f"\nClassification Report:\n"
      f"{classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0)}")

# =========================================================
# 8. Figures (displayed inline, not saved)
# =========================================================

# -- Confusion matrix --
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=CLASSES,
    cmap="Blues",
    xticks_rotation=45,
    ax=ax,
)
ax.set_title(f"Random Forest Confusion Matrix – {case_name}\n(Dataset 2: flood.csv)")
plt.tight_layout()
plt.show()

# -- Feature importance --
importances = best_model.named_steps["model"].feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(importances)), importances[indices], color="steelblue")
ax.set_xticks(range(len(importances)))
ax.set_xticklabels(np.array(feature_names)[indices], rotation=45, ha="right")
ax.set_ylabel("Mean Decrease in Impurity (Gini Importance)")
ax.set_title(f"Random Forest Feature Importance – {case_name}\n(Dataset 2: flood.csv)")
plt.tight_layout()
plt.show()

# -- ROC curves --
y_bin   = label_binarize(y_test, classes=list(range(len(CLASSES))))
y_score = best_model.predict_proba(X_test)
colors  = ["royalblue", "darkorange", "green", "red"]

fig, ax = plt.subplots(figsize=(7, 5))
for i, color in zip(range(len(CLASSES)), colors):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
    ax.plot(fpr, tpr, color=color, label=f"{CLASSES[i]} (AUC = {auc(fpr, tpr):.2f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"Random Forest ROC Curves – {case_name}\n(Dataset 2: flood.csv)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
