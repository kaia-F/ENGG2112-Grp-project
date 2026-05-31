# 🌊 FloodMap Intelligence

**Pluvial Flood Risk Prediction using Machine Learning**  
ENGG2112 Multi-disciplinary Engineering · Group 12 · University of Sydney
This Github repo is used for the appendix of the ENGG2112 group report.

---

## 📋 Overview

FloodMap Intelligence is a machine learning system that classifies urban pluvial flood susceptibility into five risk levels — **No Flood, Low, Moderate, High, and Very High** — using topographic and hydrological features. Three models are trained and compared: Random Forest, K-Nearest Neighbours (KNN), and Support Vector Machine (SVM).

The project also includes an interactive Streamlit web application that provides real-time flood risk predictions with actionable guidance for both residents and urban planners.

---

## 👥 Team

| Member | SID |
|--------|-----|
| Kaia Feng | 530289760 |
| Meri Nguyen | 550259066 |
| Minami Vaughan | 550671785 |
| Jessica Yu | 540699818 |

---

## 📁 Repository Structure

```
floodmap-intelligence/
│
├── models/
│   ├── random_forest.py       # Random Forest classifier (primary model)
│   ├── knn_model.py           # K-Nearest Neighbours classifier (baseline)
│   ├── weighted_knn.py        # Distance-weighted KNN variant
│   └── SVM.py                 # Support Vector Machine classifier
│
├── app/
│   └── floodmap_app.py        # Streamlit MVP web application
│
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🗂️ Dataset

**Pluvial Flood Dataset** — O. K. Abiodun, Kaggle, 2023.  
📎 https://www.kaggle.com/datasets/oladapokayodeabiodun/pluvial-flood-dataset

The dataset covers the **Ibadan metropolitan region, Nigeria** and contains 144,401 georeferenced spatial observations with 9 input features and 1 categorical target label.

| Feature | Description |
|---------|-------------|
| X, Y | Spatial coordinates (longitude, latitude) |
| Slope | Terrain gradient (°) |
| Curvature | Surface curvature — negative = concave (collects water) |
| Aspect | Compass direction of slope face (°) |
| TWI | Topographic Wetness Index — higher = wetter terrain |
| FA | Flow Accumulation — upstream contributing area |
| Drainage | Drainage density (km/km²) |
| Rainfall | Accumulated rainfall (mm) |
| SUSCEP | Target: No\_Flood · Low · Moderate · High · Very\_High |

> **Note:** Download the dataset from Kaggle and place `Pluvial_Flood_Dataset.xlsx` in `~/Downloads/` before running any model or the app.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/floodmap-intelligence.git
cd floodmap-intelligence

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Models

Each model script can be run independently. Toggle `INCLUDE_DRAINAGE = True/False` at the top of each file to run with or without the Drainage feature.

```bash
# Random Forest (primary model)
python models/random_forest.py

# KNN baseline
python models/knn_model.py

# Weighted KNN
python models/weighted_knn.py

# SVM
python models/SVM.py
```

Each script outputs:
- Cross-validated accuracy (10-fold)
- Macro ROC-AUC
- Full classification report
- Confusion matrix plot
- Feature importance / permutation importance plot
- ROC curves

---

## 🌐 Running the Web App

```bash
streamlit run app/floodmap_app.py
```

The app will open at `http://localhost:8501` in your browser.

### App Features

| Tab | Description |
|-----|-------------|
| 🔍 Prediction | Adjust location parameters via sliders → get real-time predictions from all 3 models with confidence scores, consensus result, stakeholder guidance, and feature driver explanation |
| 🗺️ Flood Risk Map | Interactive map of test-set predictions colour-coded by risk level. Click any point for feature details |
| 📊 Model Performance | CV accuracy, best hyperparameters, and feature descriptions |
| ℹ️ About | Project overview and risk level guide |

---

## 📊 Results Summary

All models trained on a stratified random sample of 10,000 rows with 10-fold stratified cross-validation.

### With Drainage (all 9 features)

| Model | CV Accuracy | Macro ROC-AUC | Macro F1 |
|-------|-------------|---------------|----------|
| Random Forest | **100%** | 1.000 | 1.00 |
| SVM | 90.03% | 0.993 | 0.90 |
| KNN (k=11) | 79.39% | 0.963 | 0.80 |

### Without Drainage (8 features)

| Model | CV Accuracy | Macro ROC-AUC |
|-------|-------------|---------------|
| Random Forest | 29.08% | 0.601 |
| SVM | 27.42% | 0.528 |
| KNN (k=11) | 26.58% | 0.549 |

> ⚠️ **Note on RF 100% accuracy:** Random Forest's perfect score reflects extreme Drainage dominance (92.88% Gini importance). The model learned a near-single-feature decision rule rather than a generalised classifier. SVM is the most robust model, achieving 90% accuracy with the most consistent per-class F1-scores.

---

## 🔑 Key Finding — The Drainage Feature

Drainage in this dataset refers to **drainage density** (total stream channel length per unit watershed area, km/km²) — not engineered stormwater infrastructure capacity. Higher drainage density means denser natural channel networks that respond faster to rainfall, increasing flood exposure. This is consistent with Abiodun (2020), who categorises drainage density as a top hydrological conditioning variable for pluvial flood prediction.

---

## 📦 Dependencies

See `requirements.txt`. Key packages:

```
streamlit
scikit-learn
pandas
numpy
openpyxl
matplotlib
folium
streamlit-folium
```

---

## 📚 References

- O. K. Abiodun, "Pluvial Flood Dataset," *Kaggle*, 2023. https://www.kaggle.com/datasets/oladapokayodeabiodun/pluvial-flood-dataset
- O. K. Abiodun et al., "Detection and Prediction of Pluvial Flood Using Machine Learning Techniques," *Journal of Computer Science and its Applications*, vol. 27, no. 2, Dec. 2020.

---

## 📄 Licence

This project was developed for academic purposes as part of ENGG2112 at the University of Sydney.
