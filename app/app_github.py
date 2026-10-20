import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import skfuzzy as fuzz
from safety_recommendations import get_safety_recommendations
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
def create_probability_gauge(probability, threshold, title, color):
    probability_pct = float(probability) * 100
    threshold_pct = threshold * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_pct,
            number={
                "suffix": "%",
                "valueformat": ".2f",
                "font": {"size": 30, "color": "white"},
            },
            title={
                "text": title,
                "font": {"size": 17, "color": "white"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "white",
                    "tickfont": {"color": "white"},
                },
                "bar": {"color": color, "thickness": 0.32},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, 100],
                        "color": "#303642",
                    }
                ],
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 4},
                    "thickness": 0.8,
                    "value": threshold_pct,
                },
            },
        )
    )

    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )

    return fig

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


# Load prediction models and terrain data safely
try:
    flood_model, landslide_model, features = load_models()
    terrain = load_terrain()

except FileNotFoundError as e:
    st.error(
        "Required model or dataset file is missing. "
        "Please check the models and data folders."
    )
    st.stop()

except Exception as e:
    st.error(
        "Unable to initialize the prediction system. "
        "Please check the application setup."
    )
    st.stop()

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
def generate_prediction_pdf(prediction_data):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    elements = []

    elements.append(
        Paragraph(
            "Sri Lanka Flood & Landslide Risk Prediction",
            styles["Title"]
        )
    )

    elements.append(
        Paragraph(
            "Disaster Risk Assessment Report",
            styles["Heading2"]
        )
    )

    elements.append(Spacer(1, 16))

    # Use the same prediction values as the CSV export
    table_data = [["Parameter", "Value"]]

    for key, value in prediction_data.items():
        table_data.append([str(key), str(value)])

    table = Table(
        table_data,
        colWidths=[240, 270],
        repeatRows=1,
        hAlign="LEFT"
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "Disclaimer: This is an academic prototype. "
            "Predictions do not replace official disaster warnings "
            "issued by Sri Lankan authorities.",
            styles["BodyText"]
        )
    )

    doc.build(elements)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes

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

# Default weather input values
DEFAULT_WEATHER = {
    "rainfall_24h": 20.0,
    "rainfall_72h": 50.0,
    "rainfall_7day": 100.0,
    "mean_temp": 27.0,
    "dew_temp": 23.0,
    "humidity": 80.0
}

# Initialize weather input state
for key, default in DEFAULT_WEATHER.items():
    if key not in st.session_state:
        st.session_state[key] = default


def reset_weather_inputs():
    for key, default in DEFAULT_WEATHER.items():
        st.session_state[key] = default

st.subheader("🌦️ Weather Information")

col1, col2, col3 = st.columns(3)

with col1:

    rainfall_24h = st.number_input(
    "Rainfall - Last 24 Hours (mm)",
    min_value=0.0,
    key="rainfall_24h",
    step=1.0
)

    mean_temp = st.number_input(
    "Mean Temperature (°C)",
    min_value=-10.0,
    max_value=50.0,
    key="mean_temp",
    step=0.1,
    help="Enter the mean air temperature in degrees Celsius."
)


with col2:

    rainfall_72h = st.number_input(
    "Rainfall - Last 72 Hours (mm)",
    min_value=0.0,
    key="rainfall_72h",
    step=1.0
)

    dew_temp = st.number_input(
    "Dew Point Temperature (°C)",
    min_value=-10.0,
    max_value=50.0,
    key="dew_temp",
    step=0.1,
    help="Enter the dew point temperature in degrees Celsius."
)


with col3:

    rainfall_7day = st.number_input(
    "Rainfall - Last 7 Days (mm)",
    min_value=0.0,
    key="rainfall_7day",
    step=1.0
)

    humidity = st.number_input(
    "Relative Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    key="humidity",
    step=1.0
)
    # Reset weather inputs
st.button(
    "🔄 Reset Weather Inputs",
    on_click=reset_weather_inputs,
    use_container_width=True
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

    # Check weather input consistency
    if dew_temp > mean_temp:
        st.warning(
            "The dew point temperature is higher than the mean "
            "air temperature. Please verify your weather inputs."
        )
            # Check rainfall input consistency
    if rainfall_24h > rainfall_72h:
        st.warning(
            "The 24-hour rainfall exceeds the 72-hour rainfall. "
            "Please verify your rainfall inputs."
        )

    if rainfall_72h > rainfall_7day:
        st.warning(
            "The 72-hour rainfall exceeds the 7-day rainfall. "
            "Please verify your rainfall inputs."
        )

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

        # Display prediction generation time in Sri Lanka
    prediction_time = datetime.now(
        ZoneInfo("Asia/Colombo")
    )

    st.caption(
        "🕒 Prediction generated on: "
        + prediction_time.strftime("%d %B %Y, %I:%M:%S %p")
        + " (Sri Lanka Time)"
    )

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

    # ==============================================
    # Disaster Safety Recommendations
    # ==============================================

    st.divider()
    st.subheader("🛡️ Disaster Safety Recommendations")

    col_flood, col_landslide = st.columns(2)

    with col_flood:
        st.markdown("### 🌊 Flood Safety")

        flood_tips = get_safety_recommendations(
            "flood",
            result["flood_risk"]
        )

        for tip in flood_tips:
            st.write("•", tip)

    with col_landslide:
        st.markdown("### ⛰️ Landslide Safety")

        landslide_tips = get_safety_recommendations(
            "landslide",
            result["landslide_risk"]
        )

        for tip in landslide_tips:
            st.write("•", tip)

    st.info(
        "These recommendations provide general safety "
        "guidance only. Always follow official disaster "
        "warnings from Sri Lankan authorities."
    )

        # =====================================================
    # NEW: AI PREDICTION RESULTS - PROBABILITY GAUGES
    # =====================================================

    st.divider()
    st.subheader("🧠 AI Prediction Results")

    st.caption(
        "Random Forest probabilities, fuzzy logic scores "
        "and final risk classifications."
    )

    flood_col, landslide_col = st.columns(2)

    risk_colors = {
        "LOW": "#22C55E",
        "MEDIUM": "#F59E0B",
        "HIGH": "#EF4444"
    }

    # FLOOD GAUGE
    with flood_col:
        st.plotly_chart(
            create_probability_gauge(
                probability=result["flood_probability"],
                threshold=0.70,
                title="🌊 Flood Prediction (Random Forest)",
                color="#22C55E" if result["flood_probability"] < 0.50 else (
    "#F59E0B" if result["flood_probability"] < 0.70 else "#EF4444"
)
            ),
            use_container_width=True
        )

        st.metric(
            "Flood Fuzzy Score",
            f'{result["flood_score"]:.2f}'
        )

        st.metric(
            "Final Flood Risk",
            result["flood_risk"]
        )

    # LANDSLIDE GAUGE
    with landslide_col:
        st.plotly_chart(
            create_probability_gauge(
                probability=result["landslide_probability"],
                threshold=0.80,
                title="⛰️ Landslide Prediction (Random Forest)",
                color="#22C55E" if result["landslide_probability"] < 0.50 else (
    "#F59E0B" if result["landslide_probability"] < 0.80 else "#EF4444"
)
            ),
            use_container_width=True
        )

        st.metric(
            "Landslide Fuzzy Score",
            f'{result["landslide_score"]:.2f}'
        )

        st.metric(
            "Final Landslide Risk",
            result["landslide_risk"]
        )

    st.caption(
        "Gauge values represent machine learning probabilities. "
        "Final risk classifications also consider fuzzy logic "
        "and rule-based rainfall reasoning."
    )
    
    # Prepare prediction results for CSV download
    prediction_data = {
        "Prediction Time": prediction_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "District": selected_district,
        "Rainfall 24h (mm)": rainfall_24h,
        "Rainfall 72h (mm)": rainfall_72h,
        "Rainfall 7day (mm)": rainfall_7day,
        "Mean Temperature (C)": mean_temp,
        "Dew Point Temperature (C)": dew_temp,
        "Relative Humidity (%)": humidity,
        "Mean Elevation (m)": elevation,
        "Mean Slope (degrees)": slope,
        "Flood Risk": result["flood_risk"],
        "Flood ML Probability (%)": round(
            result["flood_probability"] * 100, 2
        ),
        "Flood Fuzzy Score": result["flood_score"],
        "Landslide Risk": result["landslide_risk"],
        "Landslide ML Probability (%)": round(
            result["landslide_probability"] * 100, 2
        ),
        "Landslide Fuzzy Score": result["landslide_score"],
        "Rule-Based Rainfall Status": result["nbro_warning"],
        "Overall Disaster Risk": result["overall_risk"]
    }

        # Convert prediction data into CSV format
    prediction_df = pd.DataFrame([prediction_data])
    csv_data = prediction_df.to_csv(index=False).encode("utf-8-sig")

    # Generate PDF report
    pdf_data = generate_prediction_pdf(prediction_data)

    # Create two columns for download buttons
    col_csv, col_pdf = st.columns(2, gap="medium")

    # LEFT SIDE - CSV Download
    with col_csv:
        st.download_button(
            label="📥 Download Prediction Results (CSV)",
            data=csv_data,
            file_name="disaster_risk_prediction.csv",
            mime="text/csv",
            use_container_width=True,
            on_click="ignore"
        )

    # RIGHT SIDE - PDF Download
    with col_pdf:
        st.download_button(
            label="📄 Download Prediction Results (PDF)",
            data=pdf_data,
            file_name="disaster_risk_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            on_click="ignore"
        )

st.divider()

st.caption(
    "Academic prototype only. This system does not replace "
    "official warnings issued by Sri Lankan authorities."
)
