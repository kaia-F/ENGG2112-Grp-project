"""
ENGG2112 - Pluvial Flood Susceptibility: kNN Model
====================================================
Set INCLUDE_DRAINAGE = True  to train with all features.
Set INCLUDE_DRAINAGE = False to train without the Drainage feature.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
from sklearn.inspection import permutation_importance

# =========================================================
# Toggle this to include or exclude the Drainage feature
# =========================================================
INCLUDE_DRAINAGE = True

# =========================================================
# 1. Load dataset
# =========================================================
DATASET_PATH = "/Users/kaia/.cache/kagglehub/datasets/oladapokayodeabiodun/pluvial-flood-dataset/versions/1"
FILE_PATH = os.path.join(DATASET_PATH, "Pluvial_Flood_Dataset.xlsx")

if not os.path.exists(FILE_PATH):
    print("[INFO] Dataset not found – generating synthetic data for demonstration.")
    rng = np.random.default_rng(42)
    n = 1400
    df = pd.DataFrame({
        "Slope":          rng.uniform(0, 45, n),
        "Elevation":      rng.uniform(0, 500, n),
        "Rainfall":       rng.uniform(50, 400, n),
        "Soil_Type":      rng.choice([1, 2, 3, 4], n),
        "LULC":           rng.choice([1, 2, 3, 4, 5], n),
        "TWI":            rng.uniform(2, 20, n),
        "Drainage":       rng.uniform(0, 5, n),
        "Curvature":      rng.uniform(-5, 5, n),
        "Distance_River": rng.uniform(0, 2000, n),
        "SUSCEP":         rng.choice(["Low", "Moderate", "High", "Very High"], n),
    })
else:
    df = pd.read_excel(FILE_PATH)

df.columns = df.columns.str.strip()

# run first half of the dataset
df = df.iloc[:len(df) // 15].copy()

# =========================================================
# 2. Encode target
# =========================================================
label_encoder = LabelEncoder()
df["SUSCEP"] = label_encoder.fit_transform(df["SUSCEP"])
CLASSES = label_encoder.classes_
y = df["SUSCEP"]

# =========================================================
# 3. Select features based on boolean
# =========================================================
drop_cols = ["SUSCEP"] if INCLUDE_DRAINAGE else ["SUSCEP", "Drainage"]
X = df.drop(columns=drop_cols)
feature_names = list(X.columns)

case_name = "With Drainage" if INCLUDE_DRAINAGE else "Without Drainage"

print(f"\nRunning kNN — {case_name}")
print(f"Features: {feature_names}")
print(f"Dataset shape: {X.shape}")
print(f"Class distribution:\n{df['SUSCEP'].value_counts().rename(index=dict(enumerate(CLASSES)))}\n")

# =========================================================
# 4. Train / test split
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# 5. Pipeline and hyperparameter tuning
# =========================================================
# kNN requires StandardScaler so all features contribute equally by distance.
# Without scaling, features with larger ranges (e.g. Distance_River) would
# dominate the distance calculation over smaller-range features (e.g. TWI).
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   KNeighborsClassifier()),
])

# n_neighbors: odd values avoid ties; sqrt(n_train) ~ 33 is a common heuristic
#              but smaller values (3-11) are tested to find the bias-variance trade-off
# weights:     'distance' down-weights farther neighbours, useful for uneven densities
# p:           p=1 Manhattan distance, p=2 Euclidean distance
param_grid = {
    "model__n_neighbors": [3, 5, 7, 11],
    "model__weights":     ["uniform", "distance"],
    "model__p":           [1, 2],
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("[Tuning] Running GridSearchCV (10-fold CV)...")
gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
gs.fit(X_train, y_train)
best_model = gs.best_estimator_

print(f"Best hyperparameters: {gs.best_params_}")

# =========================================================
# 6. Evaluate
# =========================================================
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
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
# 7. Figures (displayed inline, not saved)
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
ax.set_title(f"kNN Confusion Matrix – {case_name}")
plt.tight_layout()
plt.show()

# -- Permutation importance --
# kNN has no built-in feature importance so permutation importance is used instead:
# each feature is randomly shuffled in turn and the resulting accuracy drop is measured.
result = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=10, random_state=42, scoring="accuracy"
)
sorted_idx = result.importances_mean.argsort()[::-1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    range(len(sorted_idx)),
    result.importances_mean[sorted_idx],
    yerr=result.importances_std[sorted_idx],
    color="steelblue", ecolor="black", capsize=4,
)
ax.set_xticks(range(len(sorted_idx)))
ax.set_xticklabels(np.array(feature_names)[sorted_idx], rotation=45, ha="right")
ax.set_ylabel("Mean Accuracy Drop (permutation)")
ax.set_title(f"kNN Permutation Importance – {case_name}")
plt.tight_layout()
plt.show()

# -- ROC curves --
y_bin   = label_binarize(y_test, classes=list(range(len(CLASSES))))
y_score = best_model.predict_proba(X_test)
colors  = ["royalblue", "darkorange", "green", "red", "blue"]

fig, ax = plt.subplots(figsize=(7, 5))
for i, color in zip(range(len(CLASSES)), colors):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
    ax.plot(fpr, tpr, color=color, label=f"{CLASSES[i]} (AUC = {auc(fpr, tpr):.2f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"kNN ROC Curves – {case_name}")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
