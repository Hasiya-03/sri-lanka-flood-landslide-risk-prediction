"""Sri Lanka Disaster Risk Monitor — responsive Streamlit dashboard.

Preserves the original Random Forest, fuzzy scoring, rule-based logic, and exports.
Requires the existing app/district_map.py with map_key parameter.
"""

import html
import base64
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import skfuzzy as fuzz
from district_map import show_district_map
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

st.set_page_config(
    page_title="Sri Lanka | Disaster Risk Monitor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = Path(__file__).resolve().parent / "styles.css"
if CSS_PATH.exists():
    css = CSS_PATH.read_text(encoding="utf-8")
    # Bundle scenic art directly into the stylesheet as data URIs.
    # This avoids broken relative URLs when Streamlit serves the app.
    
    
    for name, placeholder in (
        ("highland_hero.webp", "__HERO_ART__"),
        ("forest_panel.webp", "__FOREST_ART__"),
        ("3.png", "__SIDEBAR_ART__"),
        ("1.png", "__NEW_HERO_ART__"),
        ("8.png", "__SRI_LANKA_ART__"),
        ("9.png", "__DISTRICT_ART__"),
        ("4.png", "__WEATHER_ART__"),
        ("5.png", "__FLOOD_ART__"),
        ("6.png", "__LANDSLIDE_ART__"),
        ("7.png", "__OVERALL_ART__"),
        ("9.png", "__REPORTS_ART__"),
    ):
        image_path = Path(__file__).resolve().parent / "assets" / name
        if image_path.exists():
            image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
            css = css.replace(placeholder, f"data:image/{'png' if image_path.suffix.lower() == '.png' else 'webp'};base64,{image_data}")
        else:
            css = css.replace(placeholder, "none")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
else:
    st.warning("app/styles.css was not found. The app will use default Streamlit styling.")


def create_probability_gauge(probability, threshold, title, color):
    """Show the original model probability and its configured ML threshold."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(probability)*100,
        number={"suffix":"%", "valueformat":".2f", "font":{"size":23,"color":"#E8EEE9"}},
        title={"text":title,"font":{"size":12,"color":"#B0C0BB"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#82958F","tickfont":{"color":"#A4B8B1"}},
            "bar":{"color":color,"thickness":0.26},
            "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
            "steps":[{"range":[0,100],"color":"#273A3E"}],
            "threshold":{"line":{"color":"#EBE9D7","width":3},"thickness":0.8,"value":threshold*100},
        },
    ))
    fig.update_layout(height=192,margin=dict(l=5,r=5,t=40,b=0),
                      paper_bgcolor="rgba(0,0,0,0)",font={"color":"#E8EEE9"})
    return fig


def panel_heading(eyebrow, heading, detail=""):
    eyebrow = html.escape(eyebrow)
    heading = html.escape(heading)
    detail = html.escape(detail)
    st.markdown(
        f'<div class="panel-heading"><span class="eyebrow">{eyebrow}</span>'
        f'<h2>{heading}</h2><p>{detail}</p></div>',
        unsafe_allow_html=True,
    )


def risk_card(label, level, supporting):
    safe_level = html.escape(str(level).upper())
    cls = {"LOW":"low","MEDIUM":"medium","HIGH":"high"}.get(safe_level,"unknown")
    st.markdown(
        f'<div class="risk-card {cls}"><div class="risk-label">{html.escape(label)}</div>'
        f'<div class="risk-value">{safe_level}</div>'
        f'<div class="risk-support">{html.escape(supporting)}</div></div>',
        unsafe_allow_html=True,
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

    # Match the feature order saved with the trained models, if available.
    if isinstance(features, (list, tuple, pd.Index, np.ndarray)):
        trained_features = list(features)
        if trained_features and set(trained_features) == set(input_data.columns):
            input_data = input_data[trained_features]

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
# RESPONSIVE WEATHER-INTELLIGENCE DASHBOARD
# =========================================================

DEFAULT_WEATHER = {
    "rainfall_24h": 20.0,
    "rainfall_72h": 50.0,
    "rainfall_7day": 100.0,
    "mean_temp": 27.0,
    "dew_temp": 23.0,
    "humidity": 80.0,
}
for _field, _default in DEFAULT_WEATHER.items():
    st.session_state.setdefault(_field, _default)


def reset_weather_inputs():
    for field, default in DEFAULT_WEATHER.items():
        st.session_state[field] = default
    st.session_state.pop("latest_assessment", None)




# ==========================================
# DISASTERWATCH SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:
    # One HTML block keeps the navigation list together for consistent styling.
    # These are same-page anchors, not Streamlit multipage links.
    nav_items = [
        ("Dashboard", "dashboard", "⌂"),
        ("Risk Prediction", "prediction", "▥"),
        ("District Information", "district", "⌖"),
        ("Model Insights", "insights", "◫"),
        ("Reports", "reports", "▤"),
        ("About System", "about", "ⓘ"),
    ]
    links = "".join(
        f'<a class="nav-item {"active" if target == "dashboard" else ""}" '
        f'href="#{target}"><span class="nav-icon" aria-hidden="true">{icon}</span>'
        f'<span>{html.escape(label)}</span></a>'
        for label, target, icon in nav_items
    )
    st.markdown(
        """
<div class="sidebar-brand">
  <div class="brand-symbol">
    <svg viewBox="0 0 100 80" width="76" height="65" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
      <path d="M5 65 L32 25 L48 45 L65 12 L96 65 Z" fill="#83C996"/>
      <path d="M32 25 L42 40 L48 45 L39 47 Z" fill="#D5F1D5"/>
      <path d="M65 12 L78 37 L69 33 L60 38 Z" fill="#D5F1D5"/>
      <path d="M5 65 Q40 45 96 65 L96 73 L5 73 Z" fill="#256D60"/>
    </svg>
  </div>
  <h2>DisasterWatch</h2>
  <p>Sri Lanka</p>
</div>
<div class="sidebar-menu">
""" + links + """
</div>
 <div class="sidebar-bottom">
        <div class="sidebar-footer-content">
            <svg width="30" height="36" viewBox="0 0 30 36"
                 xmlns="http://www.w3.org/2000/svg"
                 aria-hidden="true">
                <path d="M15 31V13" stroke="#8FDE9E"
                      stroke-width="2" stroke-linecap="round"/>
                <path d="M15 20C5 19 3 11 7 5C15 6 19 13 15 20Z"
                      fill="#70BA83"/>
                <path d="M15 25C25 23 28 15 24 10C17 12 13 18 15 25Z"
                      fill="#A5E4A9"/>
                <path d="M15 17C9 9 7 7 7 5M15 22L24 10"
                      stroke="#285F47" stroke-width="1.2"/>
            </svg>
            <p>AI-powered<br>
            Flood &amp; Landslide<br>
            Risk Prediction System</p>
        </div>
        <div class="sidebar-progress">
            <div class="sidebar-progress-fill"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Scoped navigation styles complement the existing app/styles.css.
st.markdown("""
<style>
section[data-testid="stSidebar"] .sidebar-menu {display:flex;flex-direction:column;gap:5px;margin-top:10px}
section[data-testid="stSidebar"] .nav-item {display:flex;align-items:center;gap:14px;min-height:44px;padding:11px 15px;border-radius:9px;color:#D8E8E3!important;text-decoration:none!important;font-weight:500;font-size:14px}
section[data-testid="stSidebar"] .nav-item:hover {background:rgba(79,151,123,.22)}
section[data-testid="stSidebar"] .nav-item.active {background:linear-gradient(90deg,#287557,#175246);border-left:3px solid #AFE8B0;color:white!important}
section[data-testid="stSidebar"] .nav-icon {display:inline-flex;width:21px;align-items:center;justify-content:center;font-size:20px;line-height:1;color:#B9E4CD}
@media(max-width:768px){section[data-testid="stSidebar"] .nav-item{min-height:46px}}
</style>
""", unsafe_allow_html=True)

st.markdown('<div id="dashboard"></div>', unsafe_allow_html=True)






# ==========================================
# DISASTERWATCH CINEMATIC HERO
# ==========================================

from textwrap import dedent

hero_html = dedent("""
<div class="hero hero-premium">
    <div class="hero-main">
        <div class="hero-eyebrow">
            ENVIRONMENTAL INTELLIGENCE PLATFORM
        </div>

        <h1>Sri Lanka DisasterWatch</h1>

        <p class="hero-subtitle">
            Flood and Landslide Risk Prediction using
            Machine Learning and Environmental Data
        </p>

        <div class="hero-underline"></div>

        <p class="hero-description">
            Analyze environmental conditions, terrain
            information and machine learning models to
            assess flood and landslide risk across
            Sri Lankan districts.
        </p>
    </div>

    <div class="hero-country-card">
        <div class="hero-country-image"></div>

        <div class="hero-country-copy">
            <span>SRI LANKA</span>
            <div class="country-underline"></div>
            <p>
                Safer Communities<br>
                Through Data<br>
                and AI
            </p>
        </div>
    </div>
</div>
""").strip()

st.html(hero_html)





# =========================================================
# DISTRICT OVERVIEW & DISTRICT INFORMATION
# =========================================================

st.html('<div id="district"></div>')

districts = sorted(terrain["District"].dropna().unique())

overview_col, info_col = st.columns([1, 1], gap="small")


with overview_col:
    with st.container(border=True, key="district_overview"):

        title_col, select_col = st.columns(
            [1.6, 1],
            gap="small",
            vertical_alignment="center"
        )

        with title_col:
            st.html("""
<div class="district-panel-heading">
    <span class="district-heading-icon">⌖</span>
    <div>
        <h3>District Overview</h3>
        <p>Select a district to view its location on the map.</p>
    </div>
</div>
""")

        with select_col:
            selected_district = st.selectbox(
                "Select District",
                districts,
                format_func=lambda d: str(d).title(),
                key="selected_district"
            )

        district_info = terrain.loc[
            terrain["District"] == selected_district
        ].iloc[0]

        elevation = float(district_info["Mean_Elevation_m"])
        slope = float(district_info["Mean_Slope_deg"])

        show_district_map(
            selected_district,
            map_key="ui_overview_map",
            map_height=260
        )


with info_col:
    with st.container(border=True, key="district_information"):

        st.html("""
<div class="district-panel-heading">
    <span class="district-heading-icon">▥</span>
    <div>
        <h3>District Information</h3>
        <p>Key terrain characteristics of the selected district.</p>
    </div>
</div>
""")

        st.metric("Elevation", f"{elevation:.1f} m")
        st.metric("Mean Slope", f"{slope:.1f}°")

        st.caption(
            "Terrain measurements are district-level averages "
            "from the project dataset."
        )

# =========================================================
# ENVIRONMENTAL INPUTS CONTINUE BELOW
# =========================================================



# =========================================================
# ENVIRONMENTAL DATA — PREMIUM DASHBOARD
# =========================================================

st.html('<div id="prediction"></div>')

with st.container(border=True, key="environmental_data"):

    st.html("""
<div class="environment-heading">
    <span class="environment-icon">☁</span>
    <div>
        <h3>Environmental Data</h3>
        <p>Enter rainfall and atmospheric measurements to generate a disaster risk prediction.</p>
    </div>
</div>
""")

    rainfall_col, atmosphere_col, actions_col = st.columns(
        [4.2, 4.2, 2.3],
        gap="small",
        vertical_alignment="bottom"
    )
    with rainfall_col:
        with st.container(border=True, key="rainfall_panel"):
            st.markdown("**Rainfall Measurements**")

            rain1, rain2, rain3 = st.columns(3, gap="small")

            with rain1:
                rainfall_24h = st.number_input(
                    "Last 24 Hours (mm)",
                    min_value=0.0,
                    step=1.0,
                    key="rainfall_24h"
                )

            with rain2:
                rainfall_72h = st.number_input(
                    "Last 72 Hours (mm)",
                    min_value=0.0,
                    step=1.0,
                    key="rainfall_72h"
                )

            with rain3:
                rainfall_7day = st.number_input(
                    "Last 7 Days (mm)",
                    min_value=0.0,
                    step=1.0,
                    key="rainfall_7day"
                )

    with atmosphere_col:
        with st.container(border=True, key="atmosphere_panel"):
            st.markdown("**Atmospheric Conditions**")

            atm1, atm2, atm3 = st.columns(3, gap="small")

            with atm1:
                mean_temp = st.number_input(
                    "Mean Temperature (°C)",
                    min_value=-10.0,
                    max_value=50.0,
                    step=0.1,
                    key="mean_temp"
                )

            with atm2:
                dew_temp = st.number_input(
                    "Dew Point (°C)",
                    min_value=-10.0,
                    max_value=50.0,
                    step=0.1,
                    key="dew_temp"
                )

            with atm3:
                humidity = st.number_input(
                    "Relative Humidity (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=1.0,
                    key="humidity"
                )

    with actions_col:
        st.button(
            "Reset Inputs",
            on_click=reset_weather_inputs,
            use_container_width=True,
            key="weather_reset_button"
        )

        run_prediction = st.button(
            "Generate Risk Assessment",
            type="primary",
            use_container_width=True,
            key="weather_predict_button"
        )

current_inputs = (str(selected_district), rainfall_24h, rainfall_72h, rainfall_7day,
                  mean_temp, dew_temp, humidity, elevation, slope)
if run_prediction:
    problems = []
    if dew_temp > mean_temp:
        problems.append("Dew point exceeds mean temperature. Check your atmospheric measurements.")
    if rainfall_24h > rainfall_72h:
        problems.append("24-hour accumulated rainfall cannot exceed the 72-hour total for the same period.")
    if rainfall_72h > rainfall_7day:
        problems.append("72-hour accumulated rainfall cannot exceed the 7-day total for the same period.")
    if problems:
        for issue in problems:
            st.warning(issue)
        st.session_state.pop("latest_assessment", None)
    else:
        prediction = predict_risk(rainfall_24h,rainfall_72h,rainfall_7day,
                                  mean_temp,dew_temp,humidity,elevation,slope)
        timestamp = datetime.now(ZoneInfo("Asia/Colombo"))
        st.session_state["latest_assessment"] = {
            "inputs": current_inputs,
            "result": prediction,
            "prediction_time": timestamp,
        }


# Sidebar anchors are available even before the first assessment.
st.html('<div id="insights"></div><div id="reports"></div>')

assessment = st.session_state.get("latest_assessment")

if assessment and assessment["inputs"] != current_inputs:
    st.info(
        "Your district or weather inputs have changed. "
        "Generate a new assessment to update the results."
    )
    assessment = None

if assessment:
    result = assessment["result"]
    prediction_time = assessment["prediction_time"]

    # =====================================================
    # PREDICTION RESULTS AND PREDICTED DISASTER RISK MAP
    # =====================================================

    overall = result["overall_risk"]


    risk_messages = {
        "LOW": "Current inputs indicate relatively low estimated flood and landslide risk.",
        "MEDIUM": "Current inputs indicate elevated disaster risk. Monitor official warnings.",
        "HIGH": "Current inputs indicate high disaster risk. Follow official safety instructions."
    }

    def risk_colour(level):
        return {
            "LOW": "#80E49C",
            "MEDIUM": "#FFD166",
            "HIGH": "#FF756E"
        }.get(str(level).upper(), "#FFFFFF")

    prediction_col, risk_map_col = st.columns(
        [1.65, 1],
        gap="small"
    )

    # -----------------------------------------------------
    # LEFT: PREDICTION RESULTS
    # -----------------------------------------------------

    with prediction_col:
        with st.container(border=True, key="prediction_results_panel"):

            st.html("""
<div class="prediction-panel-heading">
    <span class="prediction-heading-icon">&#9638;</span>
    <div>
        <h3>Prediction Results</h3>
        <p>AI-generated disaster risk assessment for the selected district.</p>
    </div>
</div>
""")

            flood_col, landslide_col, overall_col = st.columns(
                3, gap="small"
            )

            with flood_col:
                st.html(f"""
<div class="visual-risk-card flood-risk-card">
    <div class="visual-risk-title">Flood Risk</div>
    <div class="visual-risk-level"
         style="color:{risk_colour(result['flood_risk'])}">
        {html.escape(result['flood_risk'])}
    </div>
    <div class="visual-risk-details">
        <div>ML Probability:
            <strong>{result['flood_probability'] * 100:.2f}%</strong>
        </div>
        <div>Fuzzy Risk Score:
            <strong>{result['flood_score']:.2f}</strong>
        </div>
        <div>Final Classification:
            <strong>{html.escape(result['flood_risk'])}</strong>
        </div>
    </div>
</div>
""")

            with landslide_col:
                st.html(f"""
<div class="visual-risk-card landslide-risk-card">
    <div class="visual-risk-title">Landslide Risk</div>
    <div class="visual-risk-level"
         style="color:{risk_colour(result['landslide_risk'])}">
        {html.escape(result['landslide_risk'])}
    </div>
    <div class="visual-risk-details">
        <div>ML Probability:
            <strong>{result['landslide_probability'] * 100:.2f}%</strong>
        </div>
        <div>Fuzzy Risk Score:
            <strong>{result['landslide_score']:.2f}</strong>
        </div>
        <div>Final Classification:
            <strong>{html.escape(result['landslide_risk'])}</strong>
        </div>
    </div>
</div>
""")

            with overall_col:
                st.html(f"""
<div class="visual-risk-card overall-risk-card">
    <div class="visual-risk-title">Overall Risk</div>
    <div class="visual-risk-level"
         style="color:{risk_colour(overall)}">
        {html.escape(overall)}
    </div>
    <p class="visual-risk-summary">
        {html.escape(risk_messages.get(overall, "Model-generated assessment."))}
    </p>
</div>
""")

            st.caption(
                f"Assessment generated on "
                f"{prediction_time:%d %B %Y, %I:%M %p} "
                "(Sri Lanka Time)"
            )

    # -----------------------------------------------------
    # RIGHT: PREDICTED DISASTER RISK MAP
    # -----------------------------------------------------

    with risk_map_col:
        with st.container(border=True, key="predicted_risk_map_panel"):

            st.html("""
<div class="prediction-panel-heading">
    <span class="prediction-heading-icon">&#9673;</span>
    <div>
        <h3>Predicted Disaster Risk Map</h3>
        <p>Predicted flood and landslide risk at the selected district marker.</p>
    </div>
</div>
""")

            show_district_map(
                selected_district,
                flood_risk=result["flood_risk"],
                landslide_risk=result["landslide_risk"],
                map_key="ui_prediction_map",
                map_height=260
            )

            st.html("""
<div class="risk-map-legend">
    <span><i class="legend-high"></i> High Risk</span>
    <span><i class="legend-medium"></i> Medium Risk</span>
    <span><i class="legend-low"></i> Low Risk</span>
</div>
""")

            st.caption(
                "The marker represents an approximate district centre, "
                "not a mapped hazard zone."
            )


    
    # ==========================================
    # MODEL INSIGHTS — PREMIUM ANALYTICS
    # ==========================================

    insights_col, reports_col = st.columns(
        [1.8, 1],
        gap="small"
    )

    with insights_col:
        with st.container(border=True, key="model_insights_panel"):

            st.html("""
<div class="insights-heading">
    <span class="insights-icon">&#9783;</span>
    <div>
        <h3>Model Insights</h3>
        <p>Machine learning probabilities and fuzzy risk analysis.</p>
    </div>
</div>
""")

            gauge1, gauge2 = st.columns(2, gap="small")

            with gauge1:
                with st.container(key="flood_gauge_panel"):
                    st.plotly_chart(
                        create_probability_gauge(
                            result["flood_probability"],
                            0.70,
                            "Flood Probability",
                            "#88D7A0"
                        ),
                        use_container_width=True,
                        key="ui_flood_gauge"
                    )

            with gauge2:
                with st.container(key="landslide_gauge_panel"):
                    st.plotly_chart(
                        create_probability_gauge(
                            result["landslide_probability"],
                            0.80,
                            "Landslide Probability",
                            "#88D7A0"
                        ),
                        use_container_width=True,
                        key="ui_landslide_gauge"
                    )

            with st.container(key="fuzzy_scores_panel"):
                st.markdown("**Fuzzy Logic Scores**")
                fuzzy_flood, fuzzy_landslide = st.columns(2, gap="small")
                with fuzzy_flood:
                    st.caption(f'Flood: {result["flood_score"]:.2f} / 100')
                    st.progress(min(1.0, max(0.0, result["flood_score"] / 100)))
                with fuzzy_landslide:
                    st.caption(f'Landslide: {result["landslide_score"]:.2f} / 100')
                    st.progress(min(1.0, max(0.0, result["landslide_score"] / 100)))

            st.markdown(
                "**Rule-Based Rainfall Status:** "
                f"{result['nbro_warning']}"
            )

            st.caption(
                "Final risk classifications combine machine learning, "
                "fuzzy logic, and rule-based reasoning."
            )

    # The Export Reports panel is placed in reports_col below.

    
    prediction_data = {
        "Prediction Time": prediction_time.strftime("%Y-%m-%d %H:%M:%S"),
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
        "Flood ML Probability (%)": round(result["flood_probability"]*100,2),
        "Flood Fuzzy Score": result["flood_score"],
        "Landslide Risk": result["landslide_risk"],
        "Landslide ML Probability (%)": round(result["landslide_probability"]*100,2),
        "Landslide Fuzzy Score": result["landslide_score"],
        "Rule-Based Rainfall Status": result["nbro_warning"],
        "Overall Disaster Risk": result["overall_risk"],
    }
    with reports_col:
        with st.container(border=True, key="export_reports_panel"):
            st.html("""
<div class="reports-heading">
    <span class="reports-icon">&#8681;</span>
    <div>
        <h3>Export Reports</h3>
        <p>Download your disaster risk assessment for documentation.</p>
    </div>
</div>
""")

            export1, export2 = st.columns(2, gap="small")

            with export1:
                st.download_button(
                    "Download CSV",
                    data=pd.DataFrame([prediction_data]).to_csv(index=False).encode("utf-8-sig"),
                    file_name="disaster_risk_prediction.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="export_csv",
                    on_click="ignore"
                )

            with export2:
                st.download_button(
                    "Download PDF",
                    data=generate_prediction_pdf(prediction_data),
                    file_name="disaster_risk_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="export_pdf",
                    on_click="ignore"
                )

else:
    st.markdown('<div class="empty-state"><span class="eyebrow">ASSESSMENT READY</span><h3>Generate a risk assessment</h3><p>Configure the environmental inputs above to view risk scores, model insights, geographic risk analysis and report downloads.</p></div>',unsafe_allow_html=True)

# =========================================================
# ABOUT SYSTEM — accessible from the sidebar navigation
# =========================================================
st.html("""
<section id="about" class="about-section" aria-labelledby="about-title">
    <div class="about-heading">
        <div class="about-icon" aria-hidden="true">ⓘ</div>
        <div>
            <span class="about-eyebrow">ABOUT THE PLATFORM</span>
            <h2 id="about-title">About DisasterWatch</h2>
            <p>Environmental intelligence for flood and landslide risk assessment in Sri Lanka.</p>
        </div>
    </div>

    <p class="about-intro">
        DisasterWatch is an academic research prototype that combines weather measurements,
        district-level terrain data, machine-learning predictions, fuzzy logic, and rainfall-based
        reasoning to estimate flood and landslide risk for a selected Sri Lankan district.
    </p>

    <div class="about-grid">
        <article class="about-card">
            <div class="about-card-icon" aria-hidden="true">☁</div>
            <h3>Environmental Inputs</h3>
            <p>Uses rainfall over 24 hours, 72 hours and 7 days, temperature,
               dew point, humidity, and district terrain characteristics.</p>
        </article>
        <article class="about-card">
            <div class="about-card-icon" aria-hidden="true">◈</div>
            <h3>Prediction Approach</h3>
            <p>Combines two Random Forest prediction models with fuzzy risk scores
               and a rainfall-based landslide warning rule.</p>
        </article>
        <article class="about-card">
            <div class="about-card-icon" aria-hidden="true">⌖</div>
            <h3>District Risk Map</h3>
            <p>Displays the selected district's approximate centre and a marker
               coloured by its estimated risk classification.</p>
        </article>
        <article class="about-card">
            <div class="about-card-icon" aria-hidden="true">▤</div>
            <h3>Insights &amp; Reports</h3>
            <p>Shows model probabilities, fuzzy scores, rule-based rainfall status,
               and downloadable CSV and PDF assessment reports.</p>
        </article>
    </div>

    <div class="about-flow" aria-label="System workflow">
        <div class="about-flow-title">HOW IT WORKS</div>
        <div class="about-flow-steps">
            <span>01 &nbsp; Select district</span>
            <span class="about-flow-arrow" aria-hidden="true">→</span>
            <span>02 &nbsp; Enter weather data</span>
            <span class="about-flow-arrow" aria-hidden="true">→</span>
            <span>03 &nbsp; Generate assessment</span>
            <span class="about-flow-arrow" aria-hidden="true">→</span>
            <span>04 &nbsp; Review &amp; export</span>
        </div>
    </div>

    <div class="about-disclaimer">
        <strong>Important:</strong> DisasterWatch provides model-generated estimates for
        academic and informational purposes. It is not a live early-warning service and
        must not replace official disaster alerts, evacuation instructions, or advice
        from Sri Lankan authorities. District markers are not mapped hazard zones.
    </div>
</section>
""")

st.markdown('<div class="footer-note">ACADEMIC RESEARCH PROTOTYPE · Model-generated assessments are not official disaster warnings. Follow Sri Lankan disaster-management authorities for public safety decisions.</div>',unsafe_allow_html=True)