"""
FloodMap Intelligence — MVP Demo
ENGG2112 Group 12 (Thu 2pm)

Run with:
    pip install streamlit scikit-learn pandas numpy openpyxl folium streamlit-folium
    streamlit run floodmap_app.py
"""

import os, sys
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.set_page_config(page_title="FloodMap Intelligence", page_icon="🌊", layout="wide")

# ── Light Coral Theme CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #fffaf7 0%, #fff5f2 100%);
        color: #4a3b3b;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff0eb 0%, #ffe5de 100%);
        border-right: 1px solid #f5b7a8;
        color: #4a3b3b;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        color: #4a3b3b !important;
    }

    /* Sidebar headings */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e76f51 !important;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #ffb5a7 0%, #f4978e 100%);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
        border: 1px solid #e76f51;
        box-shadow: 0 6px 20px rgba(231, 111, 81, 0.15);
    }

    .app-header h1 {
        color: #ffffff !important;
    }

    .app-header p {
        color: #fff7f5 !important;
    }

    /* Prediction cards */
    .risk-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 2px solid #ffd6cc;
        margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(231, 111, 81, 0.08);
    }

    .risk-label {
        font-size: 13px;
        color: #a56b5d;
        margin-bottom: 6px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .risk-value {
        font-size: 26px;
        font-weight: 800;
    }

    /* Risk colours */
    .risk-No_Flood  { color: #4fc3f7; }
    .risk-Low       { color: #66bb6a; }
    .risk-Moderate  { color: #ffb74d; }
    .risk-High      { color: #ff8a65; }
    .risk-Very_High { color: #e76f51; }

    /* Confidence bars */
    .conf-name {
        width: 90px;
        font-size: 13px;
        color: #7a5a52;
    }

    .conf-bar-bg {
        flex: 1;
        background: #ffe5de;
        border-radius: 6px;
        height: 18px;
        overflow: hidden;
    }

    .conf-bar {
        height: 100%;
        border-radius: 6px;
    }

    .conf-pct {
        width: 50px;
        text-align: right;
        font-size: 13px;
        color: #7a5a52;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #8c5f55;
    }

    .stTabs [aria-selected="true"] {
        color: #e76f51 !important;
        font-weight: 600;
    }

    /* Metric badges */
    .metric-badge {
        background: #ffffff;
        border: 1px solid #ffd6cc;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        font-size: 13px;
        margin-bottom: 8px;
        box-shadow: 0 2px 8px rgba(231, 111, 81, 0.06);
    }

    .metric-badge b {
        font-size: 22px;
        color: #e76f51;
        display: block;
        margin-bottom: 4px;
    }

    /* Placeholder box */
    .placeholder-box {
        background: #fff7f4;
        border: 2px dashed #f5b7a8;
        border-radius: 12px;
        padding: 48px;
        text-align: center;
        color: #b07a6f;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f4978e 0%, #e76f51 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #e76f51 0%, #d65a3c 100%);
        color: white;
    }

    /* Dataframe and containers */
    div[data-testid="stDataFrame"],
    div[data-testid="stVerticalBlock"] > div:has(.element-container) {
        color: #4a3b3b;
    }
</style>
""", unsafe_allow_html=True)

CLASS_COLORS = {
    "No_Flood":  "#4fc3f7",
    "Low":       "#4caf50",
    "Moderate":  "#ffb300",
    "High":      "#ff7043",
    "Very_High": "#e53935",
}
CONF_BAR_COLORS = ["#4fc3f7", "#4caf50", "#ffb300", "#ff7043", "#e53935"]

ACTION_MESSAGES = {
    "No_Flood": {
        "emoji": "✅",
        "title": "No Flood Risk Detected",
        "residents": "This area shows no current flood susceptibility. No immediate action required. Continue routine monitoring of local weather and drainage conditions.",
        "planners": "This location is suitable for standard development. No flood mitigation infrastructure required at this time. Maintain regular drainage inspections.",
        "bg": "#f0fff4", "border": "#4caf50", "text": "#1b5e20",
    },
    "Low": {
        "emoji": "🟢",
        "title": "Low Flood Risk",
        "residents": "Stay informed via local weather services during heavy rainfall. Ensure household drains and gutters are clear. No evacuation necessary.",
        "planners": "Minor flood considerations apply. Incorporate basic stormwater management in development plans. Monitor drainage performance during peak rainfall seasons.",
        "bg": "#f1f8e9", "border": "#8bc34a", "text": "#33691e",
    },
    "Moderate": {
        "emoji": "⚠️",
        "title": "Moderate Flood Risk — Stay Alert",
        "residents": "Monitor Bureau of Meteorology warnings closely. Prepare an emergency kit with documents, water and medications. Identify your nearest evacuation route before an event occurs.",
        "planners": "Flood mitigation measures should be incorporated into development approvals. Consider elevated floor levels, improved drainage capacity, and community flood education programs.",
        "bg": "#fff8e1", "border": "#ffb300", "text": "#e65100",
    },
    "High": {
        "emoji": "🚨",
        "title": "High Flood Risk — Take Precautions",
        "residents": "Move valuables and furniture to higher ground now. Prepare to evacuate at short notice. Contact your local council for sandbag access. Never drive through floodwaters.",
        "planners": "This area requires active flood risk management. Restrict vulnerable land uses, mandate flood-resilient construction, and prioritise this zone in emergency response planning and early warning systems.",
        "bg": "#fff3e0", "border": "#ff7043", "text": "#bf360c",
    },
    "Very_High": {
        "emoji": "🆘",
        "title": "Very High Flood Risk — Evacuate Immediately",
        "residents": "Evacuate to your nearest designated safe refuge immediately. Call 000 if you require assistance. Do not return home until authorities declare it safe.",
        "planners": "Development in this zone should be avoided or strictly prohibited. Immediate investment in flood defence infrastructure is recommended. Prioritise community relocation programs for existing residents.",
        "bg": "#ffebee", "border": "#e53935", "text": "#b71c1c",
    },
}

def risk_emoji(label):
    return {"No_Flood": "🔵", "Low": "🟢", "Moderate": "🟡",
            "High": "🟠", "Very_High": "🔴"}.get(label, "⚪")

FEATURE_COLS = ["X", "Y", "Slope", "Curvature", "Aspect", "TWI", "FA", "Drainage", "Rainfall"]

# ── Dataset paths to try (in order) ──────────────────────────────────────────
DATASET_SEARCH_PATHS = [
    "/Users/kaia/Downloads/Pluvial_Flood_Dataset.xlsx",
    os.path.expanduser("~/Downloads/Pluvial_Flood_Dataset.xlsx"),
    os.path.expanduser("~/.cache/kagglehub/datasets/oladapokayodeabiodun/pluvial-flood-dataset/versions/1/Pluvial_Flood_Dataset.xlsx"),
]

@st.cache_data(show_spinner=False)
def load_data():
    found_path = None
    for p in DATASET_SEARCH_PATHS:
        if os.path.exists(p):
            found_path = p
            break

    if found_path:
        print(f"[FloodMap] ✅ Dataset found at: {found_path}", flush=True)
        df = pd.read_excel(found_path)
        df.columns = df.columns.str.strip()

        # Replace common GIS NoData values with NaN
        nodata_threshold = -1e30
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].mask(df[numeric_cols] < nodata_threshold)

        # Optional: also replace common sentinel values
        df.replace([-9999, -99999, 999999999], np.nan, inplace=True)
        df = df.dropna(subset=FEATURE_COLS, how="all")
        # ── FIX: random sample instead of iloc slice so all classes are represented ──
        df = df.sample(n=min(10000, len(df)), random_state=42).reset_index(drop=True)
        source = "real"
        print(f"[FloodMap] Loaded {len(df)} rows. Class distribution:\n{df['SUSCEP'].value_counts().to_string()}", flush=True)
    else:
        print("[FloodMap] ⚠️  Dataset NOT found. Searched:", flush=True)
        for p in DATASET_SEARCH_PATHS:
            print(f"           - {p}", flush=True)
        print("[FloodMap] Falling back to synthetic data.", flush=True)

        rng = np.random.default_rng(42)
        n = 10000
        slope    = rng.uniform(0, 90, n)
        twi      = rng.uniform(2, 25, n)
        rain     = rng.uniform(50, 400, n)
        drainage = rng.uniform(50, 500, n)
        fa       = rng.uniform(0, 5000, n)
        score    = (slope/90)*0.25 + (twi/25)*0.35 + (rain/400)*0.25 + (drainage/500)*0.15
        labels   = np.where(score < 0.2, "No_Flood",
                   np.where(score < 0.4, "Low",
                   np.where(score < 0.6, "Moderate",
                   np.where(score < 0.8, "High", "Very_High"))))
        df = pd.DataFrame({
            "X": rng.uniform(3, 5, n), "Y": rng.uniform(6, 9, n),
            "Slope": slope, "Curvature": rng.uniform(-10, 10, n),
            "Aspect": rng.uniform(0, 360, n), "TWI": twi,
            "FA": fa, "Drainage": drainage, "Rainfall": rain,
            "SUSCEP": labels,
        })
        source = "synthetic"
    return df, source

@st.cache_resource(show_spinner=False)
def train_models():
    df, source = load_data()

    le = LabelEncoder()
    df["SUSCEP"] = le.fit_transform(df["SUSCEP"])
    classes = le.classes_

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    y = df["SUSCEP"]
    feature_names = list(X.columns)

    # Compute real feature stats for slider ranges
    stats = {f: {"min": float(X[f].min()), "max": float(X[f].max()),
                 "mean": float(X[f].mean()), "std": float(X[f].std())}
             for f in feature_names}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_gs = GridSearchCV(
        Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("m", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"))]),
        {"m__n_estimators": [100, 200], "m__max_depth": [None, 10], "m__min_samples_split": [2, 5]},
        cv=cv, scoring="accuracy", n_jobs=-1,
    )
    rf_gs.fit(X_train, y_train)
    print(f"[FloodMap] RF best params: {rf_gs.best_params_}, CV acc: {rf_gs.best_score_:.3f}", flush=True)

    knn_gs = GridSearchCV(
        Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                  ("m", KNeighborsClassifier())]),
        {"m__n_neighbors": [3, 5, 7, 11], "m__weights": ["uniform", "distance"], "m__p": [1, 2]},
        cv=cv, scoring="accuracy", n_jobs=-1,
    )
    knn_gs.fit(X_train, y_train)
    print(f"[FloodMap] KNN best params: {knn_gs.best_params_}, CV acc: {knn_gs.best_score_:.3f}", flush=True)

    svm_gs = GridSearchCV(
        Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                  ("m", SVC(kernel="rbf", probability=True, random_state=42))]),
        {"m__C": [0.1, 1.0, 10.0], "m__gamma": ["scale", "auto"]},
        cv=cv, scoring="accuracy", n_jobs=-1,
    )
    svm_gs.fit(X_train, y_train)
    print(f"[FloodMap] SVM best params: {svm_gs.best_params_}, CV acc: {svm_gs.best_score_:.3f}", flush=True)

    # Also store sample points from test set for the map tab
    sample_map = X_test.copy()
    sample_map["true_label"]  = le.inverse_transform(y_test)
    sample_map["rf_pred"]     = le.inverse_transform(rf_gs.best_estimator_.predict(X_test))

    return {
        "models":        {"Random Forest": rf_gs.best_estimator_,
                          "KNN":           knn_gs.best_estimator_,
                          "SVM":           svm_gs.best_estimator_},
        "cv_acc":        {"Random Forest": rf_gs.best_score_,
                          "KNN":           knn_gs.best_score_,
                          "SVM":           svm_gs.best_score_},
        "best_params":   {"Random Forest": rf_gs.best_params_,
                          "KNN":           knn_gs.best_params_,
                          "SVM":           svm_gs.best_params_},
        "classes":       classes,
        "feature_names": feature_names,
        "stats":         stats,
        "sample_map":    sample_map,
        "label_encoder": le,
        "source":        source,
    }

def confidence_bars_html(proba, classes):
    pairs = sorted(zip(classes, proba), key=lambda x: -x[1])
    color_map = {c: CONF_BAR_COLORS[i] for i, c in
                 enumerate(["No_Flood", "Low", "Moderate", "High", "Very_High"])
                 if c in classes}
    bars = ""
    for cls, p in pairs:
        color = color_map.get(cls, "#4fc3f7")
        pct = p * 100
        bars += f"""
        <div class="conf-row">
            <div class="conf-name">{cls.replace("_"," ")}</div>
            <div class="conf-bar-bg">
                <div class="conf-bar" style="width:{pct:.1f}%;background:{color};"></div>
            </div>
            <div class="conf-pct">{pct:.1f}%</div>
        </div>"""
    return bars

def get_prediction_drivers(input_row, stats, feature_names, top_n=3):
    """Compare input values to dataset mean to explain what's driving risk up or down."""
    drivers_up, drivers_down = [], []
    
    feature_explanations = {
        "Drainage":  ("drainage density", "km/km²"),
        "TWI":       ("terrain wetness", ""),
        "Rainfall":  ("rainfall", "mm"),
        "Slope":     ("slope steepness", "°"),
        "FA":        ("flow accumulation", ""),
        "Curvature": ("terrain curvature", ""),
        "Aspect":    ("slope aspect", "°"),
        "X":         ("longitude", ""),
        "Y":         ("latitude", ""),
    }

    for f in feature_names:
        if f not in stats:
            continue
        val  = input_row[f]
        mean = stats[f]["mean"]
        std  = stats[f]["std"]
        if std == 0:
            continue
        z = (val - mean) / std

        label, unit = feature_explanations.get(f, (f, ""))
        val_str = f"{val:.1f}{unit}" if unit else f"{val:.2f}"

        # Features where HIGH value = higher flood risk
        risk_increasing = ["Drainage", "TWI", "Rainfall", "FA"]
        # Features where LOW value = higher flood risk  
        risk_decreasing = ["Slope"]

        if f in risk_increasing:
            if z > 0.5:
                drivers_up.append((abs(z), f"**{label}** is elevated at {val_str}"))
            elif z < -0.5:
                drivers_down.append((abs(z), f"**{label}** is low at {val_str}"))
        elif f in risk_decreasing:
            if z < -0.5:
                drivers_up.append((abs(z), f"**{label}** is gentle at {val_str}, promoting ponding"))
            elif z > 0.5:
                drivers_down.append((abs(z), f"**{label}** is steep at {val_str}, aiding runoff"))

    drivers_up.sort(reverse=True)
    drivers_down.sort(reverse=True)
    return [d[1] for d in drivers_up[:top_n]], [d[1] for d in drivers_down[:top_n]]

# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <h1 style="margin:0;font-size:2.2rem;color:#4fc3f7;">🌊 FloodMap Intelligence</h1>
    <p style="margin:8px 0 0;color:#8ab4d4;font-size:1rem;">
        Pluvial Flood Risk Prediction · ENGG2112 Group 12 · Random Forest · KNN · SVM
    </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Training models on first run — please wait ~30–60 seconds…"):
    state = train_models()

models        = state["models"]
classes       = state["classes"]
feature_names = state["feature_names"]
cv_acc        = state["cv_acc"]
stats         = state["stats"]

if state["source"] == "synthetic":
    st.error(
        "**Dataset not found.** Check your terminal for the paths that were searched. "
        "Make sure `Pluvial_Flood_Dataset.xlsx` is in your Downloads folder.",
        icon="❌",
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📍 Location Parameters")
    st.markdown("Adjust features then click **Predict**.")
    st.markdown("---")

    def smart_slider(label, key, step=None):
        s = stats.get(key, {})
        lo  = float(s.get("min", 0))
        hi  = float(s.get("max", 100))
        mid = float(s.get("mean", (lo+hi)/2))
        if step is None:
            step = max((hi - lo) / 200, 0.01)
        mid = round(mid / step) * step
        return st.slider(label, lo, hi, mid, step)

    st.markdown("**Spatial**")
    x_coord = smart_slider("X Coordinate", "X", step=0.001)
    y_coord = smart_slider("Y Coordinate", "Y", step=0.001)

    st.markdown("**Topographic**")
    slope     = smart_slider("Slope (°)",   "Slope",     step=0.5)
    curvature = smart_slider("Curvature",   "Curvature", step=0.1)
    aspect    = smart_slider("Aspect (°)",  "Aspect",    step=1.0)
    twi       = smart_slider("TWI",         "TWI",       step=0.1)

    st.markdown("**Hydrological**")
    fa       = smart_slider("Flow Accumulation", "FA",       step=1.0)
    drainage = smart_slider("Drainage",          "Drainage", step=1.0)
    rainfall = smart_slider("Rainfall (mm)",     "Rainfall", step=1.0)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Flood Risk", use_container_width=True, type="primary")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_map, tab_models, tab_about = st.tabs(
    ["🔍 Prediction", "🗺️ Flood Risk Map", "📊 Model Performance", "ℹ️ About"]
)

# ── TAB 1: Prediction ─────────────────────────────────────────────────────────
with tab_predict:
    if not predict_btn:
        st.markdown("""
        <div class="placeholder-box">
            <div style="font-size:3rem;">🌊</div>
            <div style="font-size:1.2rem;margin-top:12px;">
                Set location parameters in the sidebar<br>and click <b>Predict Flood Risk</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        raw = {"X": x_coord, "Y": y_coord, "Slope": slope, "Curvature": curvature,
               "Aspect": aspect, "TWI": twi, "FA": fa, "Drainage": drainage, "Rainfall": rainfall}
        input_df = pd.DataFrame([{f: raw[f] for f in feature_names}])

        st.markdown("### Model Predictions")
        cols = st.columns(3)
        preds = []
        for col, (name, model) in zip(cols, models.items()):
            proba      = model.predict_proba(input_df)[0]
            pred_idx   = int(np.argmax(proba))
            pred_label = classes[pred_idx]
            preds.append(pred_label)
            with col:
                st.markdown(f"""
                <div class="risk-card">
                    <div class="risk-label">{name}</div>
                    <div class="risk-value risk-{pred_label}">
                        {risk_emoji(pred_label)} {pred_label.replace("_"," ")}
                    </div>
                    <div style="margin-top:16px;">{confidence_bars_html(proba, classes)}</div>
                    <div style="margin-top:12px;font-size:12px;color:#6a8aaa;">
                        CV Accuracy: {cv_acc[name]:.1%}
                    </div>
                </div>""", unsafe_allow_html=True)

        consensus, count = Counter(preds).most_common(1)[0]
        color = CLASS_COLORS.get(consensus, "#aaa")
        agree = "All 3 models agree ✓" if count == 3 else f"{count}/3 models agree"
        st.markdown("---")
        st.markdown(f"""
        <div style="background:#fff7f4;border:2px solid {color};
                    border-radius:12px;padding:20px;text-align:center;
                    box-shadow:0 6px 18px rgba(231,111,81,0.15);">
            <div style="font-size:13px;color:#8c5f55;letter-spacing:1px;text-transform:uppercase;">
                Consensus · {agree}
            </div>
            <div style="font-size:2.4rem;font-weight:900;color:{color};margin-top:8px;">
                {risk_emoji(consensus)} {consensus.replace("_"," ")}
            </div>
        </div>""", unsafe_allow_html=True)
        action = ACTION_MESSAGES[consensus]

        # ── Action message: two-column stakeholder view ────────────────────────
        st.markdown(f"""
        <div style="border-radius:12px;padding:20px 24px;margin-top:16px;
                    border-left:5px solid {action['border']};
                    background:{action['bg']};color:{action['text']};">
            <div style="font-size:1.3rem;font-weight:700;margin-bottom:14px;">
                {action['emoji']} {action['title']}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
                <div>
                    <div style="font-weight:600;margin-bottom:6px;">🏘️ For Residents</div>
                    <div style="font-size:0.93rem;line-height:1.7;">{action['residents']}</div>
                </div>
                <div>
                    <div style="font-weight:600;margin-bottom:6px;">🏛️ For Planners & Council</div>
                    <div style="font-size:0.93rem;line-height:1.7;">{action['planners']}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── What's driving this prediction ────────────────────────────────────
        raw_row = {f: raw[f] for f in feature_names}
        drivers_up, drivers_down = get_prediction_drivers(raw_row, stats, feature_names)

        if drivers_up or drivers_down:
            st.markdown("#### 🔍 What's driving this prediction?")
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                if drivers_up:
                    st.markdown("**⬆️ Increasing risk**")
                    for d in drivers_up:
                        st.markdown(f"- {d}")
                else:
                    st.markdown("**⬆️ Increasing risk**")
                    st.markdown("- No features significantly above average")
            with dcol2:
                if drivers_down:
                    st.markdown("**⬇️ Reducing risk**")
                    for d in drivers_down:
                        st.markdown(f"- {d}")
                else:
                    st.markdown("**⬇️ Reducing risk**")
                    st.markdown("- No features significantly below average")

        st.markdown("---")

        st.markdown("#### Input Summary")
        st.dataframe(pd.DataFrame({"Feature": feature_names,
                                   "Value": [round(raw[f], 3) for f in feature_names]}),
                     use_container_width=True, hide_index=True)

# ── TAB 2: Flood Risk Map ─────────────────────────────────────────────────────
with tab_map:
    st.markdown("### Flood Risk Map")
    st.markdown("Interactive map of sampled test-set points coloured by predicted flood risk (Random Forest). Click any point for details.")

    try:
        import folium
        from streamlit_folium import st_folium

        sample = state["sample_map"].copy()

        # The dataset X/Y are in decimal degrees for Nigeria region (~3-5°E, 6-9°N)
        # folium expects lat/lon → Y=lat, X=lon
        map_center = [float(sample["Y"].mean()), float(sample["X"].mean())]

        m = folium.Map(location=map_center, zoom_start=10,
                    tiles="CartoDB positron")

        # Colour map: folium uses hex colours
        folium_colors = {
            "No_Flood":  "#4fc3f7",
            "Low":       "#4caf50",
            "Moderate":  "#ffb300",
            "High":      "#ff7043",
            "Very_High": "#e53935",
        }

        # Sample up to 500 points so the map stays fast
        plot_sample = sample.sample(n=min(500, len(sample)), random_state=1)

        for _, row in plot_sample.iterrows():
            pred  = row["rf_pred"]
            color = folium_colors.get(pred, "#ffffff")
            folium.CircleMarker(
                location=[row["Y"], row["X"]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=folium.Popup(
                    f"""<b>RF Prediction:</b> {pred.replace("_"," ")}<br>
                    <b>True Label:</b> {row["true_label"].replace("_"," ")}<br>
                    <b>Slope:</b> {row["Slope"]:.1f}°<br>
                    <b>TWI:</b> {row["TWI"]:.2f}<br>
                    <b>Rainfall:</b> {row["Rainfall"]:.1f} mm""",
                    max_width=200
                ),
                tooltip=pred.replace("_", " "),
            ).add_to(m)

        # Legend
        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:#1a2a3a;padding:12px 16px;border-radius:10px;
                    border:1px solid #2a4a6a;font-size:13px;color:#e8eaf0;">
            <b>🌊 Flood Risk</b><br>
            <span style="color:#4fc3f7;">● No Flood</span><br>
            <span style="color:#4caf50;">● Low</span><br>
            <span style="color:#ffb300;">● Moderate</span><br>
            <span style="color:#ff7043;">● High</span><br>
            <span style="color:#e53935;">● Very High</span>
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, width="100%", height=520)

        # Class breakdown chart using st.bar_chart
        st.markdown("---")
        st.markdown("#### Predicted Class Distribution (RF, test set sample)")
        dist = plot_sample["rf_pred"].value_counts().reindex(
            ["No_Flood", "Low", "Moderate", "High", "Very_High"], fill_value=0
        )
        st.bar_chart(dist, color="#4fc3f7")

    except ImportError:
        st.warning(
            "Map requires `folium` and `streamlit-folium`. Install with:\n\n"
            "```\npip install folium streamlit-folium\n```\n\nThen restart the app.",
            icon="📦"
        )
        # Fallback: static scatter using st.map
        st.markdown("**Fallback: Basic map (install folium for the interactive version)**")
        map_df = state["sample_map"].sample(n=min(500, len(state["sample_map"])), random_state=1)
        st.map(map_df.rename(columns={"Y": "lat", "X": "lon"})[["lat", "lon"]])

# ── TAB 3: Model Performance ──────────────────────────────────────────────────
with tab_models:
    st.markdown("### Model Comparison")
    c1, c2, c3 = st.columns(3)
    for col, (name, acc) in zip([c1, c2, c3], cv_acc.items()):
        with col:
            st.markdown(f"""
            <div class="metric-badge">
                <b>{acc:.1%}</b>{name}<br>
                <span style="font-size:11px;color:#6a8aaa;">5-fold CV Accuracy</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Best Hyperparameters")
    for name, params in state["best_params"].items():
        clean = {k.split("__")[-1]: v for k, v in params.items()}
        st.markdown(f"**{name}:** `{clean}`")

    st.markdown("---")
    st.markdown("### Features Used")
    st.markdown("""
| Feature | Description |
|---------|-------------|
| X, Y | Projected spatial coordinates |
| Slope | Terrain gradient (°) |
| Curvature | Surface curvature — negative = concave, collects water |
| Aspect | Slope-facing direction (°) |
| TWI | Topographic Wetness Index |
| FA | Flow Accumulation — upstream contributing area |
| Drainage | Drainage network capacity |
| Rainfall | Accumulated rainfall (mm) |
""")

# ── TAB 4: About ──────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
### FloodMap Intelligence
**ENGG2112 Multi-disciplinary Engineering — Group 12 (Thu 2pm)**

| Member | SID |
|--------|-----|
| Kaia Feng | 530289760 |
| Meri Nguyen | 550259066 |
| Minami Vaughan | 550671785 |
| Jessica Yu | 540699818 |

---
### Problem Statement
Urban flooding causes billions in damage across Australia annually. Existing flood maps are
static and low-resolution. This project uses machine learning to produce dynamic,
data-driven flood susceptibility classifications.

### Dataset
**Pluvial Flood Dataset** (Kaggle, 144,401 rows) — spatial flood susceptibility data with
topographic and hydrological features.
Target classes: **No Flood · Low · Moderate · High · Very High**

### Approach
Three classifiers (Random Forest, KNN, SVM) trained on a **stratified random sample**
of 10,000 rows with 5-fold cross-validated GridSearchCV hyperparameter tuning.
""")
