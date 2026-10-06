
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import skfuzzy as fuzz
from pathlib import Path

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Sri Lanka Disaster Risk Prediction",
    page_icon="🌧️",
    layout="wide"
)

# =========================================================
# LOAD MODELS AND DATA
# =========================================================

# Find the project root folder.
# app.py is inside project/app/, so parents[1] points to project/.
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


@st.cache_resource
def load_models():
    flood_model = joblib.load(MODEL_DIR / "flood_random_forest.pkl")
    landslide_model = joblib.load(MODEL_DIR / "landslide_random_forest.pkl")
    features = joblib.load(MODEL_DIR / "model_features.pkl")

    return flood_model, landslide_model, features


@st.cache_data
def load_terrain():
    return pd.read_csv(DATA_DIR / "district_terrain.csv")


flood_model, landslide_model, features = load_models()
terrain = load_terrain()

# =========================================================
# FUZZY MEMBERSHIP FUNCTIONS
# =========================================================

rainfall_range = np.arange(0, 301, 1)
slope_range = np.arange(0, 61, 1)
probability_range = np.arange(0, 1.01, 0.01)

rain_low = fuzz.trapmf(
    rainfall_range, [0, 0, 50, 75]
)

rain_medium = fuzz.trimf(
    rainfall_range, [50, 100, 150]
)

rain_high = fuzz.trapmf(
    rainfall_range, [100, 150, 300, 300]
)

slope_low = fuzz.trapmf(
    slope_range, [0, 0, 5, 10]
)

slope_medium = fuzz.trimf(
    slope_range, [5, 15, 30]
)

slope_high = fuzz.trapmf(
    slope_range, [20, 30, 60, 60]
)

prob_low = fuzz.trapmf(
    probability_range, [0, 0, 0.20, 0.40]
)

prob_medium = fuzz.trimf(
    probability_range, [0.20, 0.50, 0.80]
)

prob_high = fuzz.trapmf(
    probability_range, [0.60, 0.80, 1.0, 1.0]
)


# =========================================================
# FUZZY RISK FUNCTION
# =========================================================

def calculate_fuzzy_risk(rainfall, slope, ml_probability):

    rainfall = np.clip(rainfall, 0, 300)
    slope = np.clip(slope, 0, 60)
    ml_probability = np.clip(ml_probability, 0, 1)

    r_low = fuzz.interp_membership(
        rainfall_range, rain_low, rainfall
    )

    r_medium = fuzz.interp_membership(
        rainfall_range, rain_medium, rainfall
    )

    r_high = fuzz.interp_membership(
        rainfall_range, rain_high, rainfall
    )

    s_low = fuzz.interp_membership(
        slope_range, slope_low, slope
    )

    s_medium = fuzz.interp_membership(
        slope_range, slope_medium, slope
    )

    s_high = fuzz.interp_membership(
        slope_range, slope_high, slope
    )

    p_low = fuzz.interp_membership(
        probability_range, prob_low, ml_probability
    )

    p_medium = fuzz.interp_membership(
        probability_range, prob_medium, ml_probability
    )

    p_high = fuzz.interp_membership(
        probability_range, prob_high, ml_probability
    )

    low_strength = max(
        min(r_low, s_low),
        p_low
    )

    medium_strength = max(
        r_medium,
        s_medium,
        p_medium
    )

    high_strength = max(
        r_high,
        s_high,
        p_high
    )

    total = low_strength + medium_strength + high_strength

    if total == 0:
        return 0.0

    score = (
        low_strength * 20
        + medium_strength * 55
        + high_strength * 90
    ) / total

    return round(float(score), 2)


# =========================================================
# RISK LEVEL
# =========================================================

def score_to_risk_level(score):

    if score < 40:
        return "LOW"

    elif score < 70:
        return "MEDIUM"

    else:
        return "HIGH"


# =========================================================
# RULE-BASED LANDSLIDE WARNING
# =========================================================

def nbro_landslide_rule(rainfall_24h):

    if rainfall_24h >= 150:
        return "HIGH"

    elif rainfall_24h >= 100:
        return "MEDIUM"

    elif rainfall_24h >= 75:
        return "LOW-WATCH"

    else:
        return "NORMAL"


# =========================================================
# FINAL AI SYSTEM
# =========================================================

def predict_risk(
    rainfall_24h,
    rainfall_72h,
    rainfall_7day,
    mean_temp,
    dew_temp,
    humidity,
    elevation,
    slope
):

    input_data = pd.DataFrame([{
        "Rainfall_24h": rainfall_24h,
        "Rainfall_72h": rainfall_72h,
        "Rainfall_7day": rainfall_7day,
        "Mean_Temp_C": mean_temp,
        "Dew_Temp_C": dew_temp,
        "Relative_Humidity": humidity,
        "Mean_Elevation_m": elevation,
        "Mean_Slope_deg": slope
    }])

    # Machine Learning
    flood_probability = flood_model.predict_proba(
        input_data
    )[0][1]

    landslide_probability = landslide_model.predict_proba(
        input_data
    )[0][1]

    # Fuzzy Logic
    flood_score = calculate_fuzzy_risk(
        rainfall_24h,
        slope,
        flood_probability
    )

    landslide_score = calculate_fuzzy_risk(
        rainfall_24h,
        slope,
        landslide_probability
    )

    flood_risk = score_to_risk_level(flood_score)
    landslide_risk = score_to_risk_level(landslide_score)

    # ML validation thresholds
    if flood_probability >= 0.70:
        flood_risk = "HIGH"

    elif flood_probability >= 0.50 and flood_risk == "LOW":
        flood_risk = "MEDIUM"

    if landslide_probability >= 0.80:
        landslide_risk = "HIGH"

    elif landslide_probability >= 0.50 and landslide_risk == "LOW":
        landslide_risk = "MEDIUM"

    # Rule-based reasoning
    nbro_warning = nbro_landslide_rule(rainfall_24h)

    if nbro_warning == "HIGH":
        landslide_risk = "HIGH"

    elif nbro_warning == "MEDIUM" and landslide_risk == "LOW":
        landslide_risk = "MEDIUM"

    # Overall risk
    risk_order = {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3
    }

    overall_risk = max(
        [flood_risk, landslide_risk],
        key=lambda x: risk_order[x]
    )

    return {
        "flood_probability": float(flood_probability),
        "landslide_probability": float(landslide_probability),

        "flood_score": flood_score,
        "landslide_score": landslide_score,

        "flood_risk": flood_risk,
        "landslide_risk": landslide_risk,

        "nbro_warning": nbro_warning,
        "overall_risk": overall_risk
    }


# =========================================================
# USER INTERFACE
# =========================================================

st.title("🌧️ Sri Lanka Flood & Landslide Risk Prediction")

st.write(
    "AI-based prototype combining Machine Learning, "
    "Fuzzy Logic and Rule-Based Reasoning."
)

st.divider()

# -------------------------
# District
# -------------------------

districts = sorted(terrain["District"].unique())

selected_district = st.selectbox(
    "📍 Select District",
    districts
)

district_info = terrain[
    terrain["District"] == selected_district
].iloc[0]

elevation = float(
    district_info["Mean_Elevation_m"]
)

slope = float(
    district_info["Mean_Slope_deg"]
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Mean Elevation",
        f"{elevation:.1f} m"
    )

with col2:
    st.metric(
        "Mean Slope",
        f"{slope:.1f}°"
    )


st.subheader("🌦️ Weather Information")

col1, col2, col3 = st.columns(3)

with col1:

    rainfall_24h = st.number_input(
        "Rainfall - Last 24 Hours (mm)",
        min_value=0.0,
        value=20.0,
        step=1.0
    )

    mean_temp = st.number_input(
        "Mean Temperature (°C)",
        value=27.0,
        step=0.1
    )


with col2:

    rainfall_72h = st.number_input(
        "Rainfall - Last 72 Hours (mm)",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    dew_temp = st.number_input(
        "Dew Point Temperature (°C)",
        value=23.0,
        step=0.1
    )


with col3:

    rainfall_7day = st.number_input(
        "Rainfall - Last 7 Days (mm)",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

    humidity = st.number_input(
        "Relative Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )


st.divider()

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔍 Predict Disaster Risk",
    type="primary",
    use_container_width=True
):

    result = predict_risk(
        rainfall_24h,
        rainfall_72h,
        rainfall_7day,
        mean_temp,
        dew_temp,
        humidity,
        elevation,
        slope
    )

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "🌊 Flood Risk",
            result["flood_risk"]
        )

        st.write(
            "ML Probability:",
            f'{result["flood_probability"] * 100:.2f}%'
        )

        st.write(
            "Fuzzy Risk Score:",
            result["flood_score"]
        )


    with col2:

        st.metric(
            "⛰️ Landslide Risk",
            result["landslide_risk"]
        )

        st.write(
            "ML Probability:",
            f'{result["landslide_probability"] * 100:.2f}%'
        )

        st.write(
            "Fuzzy Risk Score:",
            result["landslide_score"]
        )


    st.divider()

    st.subheader("🚨 Overall Disaster Risk")

    overall = result["overall_risk"]

    if overall == "HIGH":

        st.error(
            "🔴 HIGH RISK"
        )

        st.warning(
            "Potential hazardous conditions detected. "
            "Follow official disaster-management and "
            "local authority warnings."
        )

    elif overall == "MEDIUM":

        st.warning(
            "🟠 MEDIUM RISK"
        )

        st.info(
            "Conditions indicate increased risk. "
            "Continue monitoring rainfall and official warnings."
        )

    else:

        st.success(
            "🟢 LOW RISK"
        )

        st.info(
            "Current inputs indicate relatively low risk, "
            "but conditions may change."
        )


    st.write(
        "**Rule-Based Rainfall Status:**",
        result["nbro_warning"]
    )


st.divider()

st.caption(
    "Academic prototype only. This system does not replace "
    "official warnings issued by Sri Lankan authorities."
)
