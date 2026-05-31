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
