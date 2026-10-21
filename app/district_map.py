"""Interactive district-centre map for Sri Lanka DisasterWatch.

The marker shows an approximate district centre, not a hazard-zone polygon.
"""

from pathlib import Path
import html
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium

DISTRICT_COORDINATES = {
    "ampara": (7.30, 81.67),
    "anuradhapura": (8.31, 80.40),
    "badulla": (6.99, 81.06),
    "batticaloa": (7.72, 81.70),
    "colombo": (6.93, 79.86),
    "galle": (6.05, 80.22),
    "gampaha": (7.09, 79.99),
    "hambantota": (6.12, 81.12),
    "jaffna": (9.66, 80.02),
    "kalutara": (6.59, 79.96),
    "kandy": (7.29, 80.64),
    "kegalle": (7.25, 80.35),
    "kilinochchi": (9.38, 80.40),
    "kurunegala": (7.49, 80.36),
    "mannar": (8.98, 79.90),
    "matale": (7.47, 80.62),
    "matara": (5.95, 80.55),
    "monaragala": (6.87, 81.35),
    "mullaitivu": (9.27, 80.81),
    "nuwara eliya": (6.97, 80.77),
    "polonnaruwa": (7.94, 81.00),
    "puttalam": (8.04, 79.84),
    "ratnapura": (6.68, 80.40),
    "trincomalee": (8.57, 81.23),
    "vavuniya": (8.75, 80.50),
}


@st.cache_data(show_spinner=False)
def _load_district_terrain():
    data_path = Path(__file__).resolve().parent.parent / "data" / "district_terrain.csv"
    data = pd.read_csv(data_path)
    data["District"] = data["District"].astype(str).str.lower().str.strip()
    return data


def show_district_map(
    selected_district,
    flood_risk=None,
    landslide_risk=None,
    map_key="district_terrain_map",
    map_height=450,
):
    """Show an interactive district-centre pin. Risk affects marker color only."""
    district = str(selected_district).lower().strip()
    if district not in DISTRICT_COORDINATES:
        st.warning("Location not available for this district.")
        return

    latitude, longitude = DISTRICT_COORDINATES[district]
    data = _load_district_terrain()
    district_row = data.loc[data["District"] == district]

    popup_parts = [f"<b>District:</b> {html.escape(district.title())}"]
    if not district_row.empty:
        row = district_row.iloc[0]
        elevation = pd.to_numeric(row.get("Mean_Elevation_m"), errors="coerce")
        slope = pd.to_numeric(row.get("Mean_Slope_deg"), errors="coerce")
        if pd.notna(elevation):
            popup_parts.append(f"<b>Mean Elevation:</b> {elevation:.2f} m")
        if pd.notna(slope):
            popup_parts.append(f"<b>Mean Slope:</b> {slope:.2f}°")

    if flood_risk is not None:
        popup_parts.append(f"<b>Flood Risk:</b> {html.escape(str(flood_risk))}")
    if landslide_risk is not None:
        popup_parts.append(f"<b>Landslide Risk:</b> {html.escape(str(landslide_risk))}")

    risk_levels = {
        str(level).upper()
        for level in (flood_risk, landslide_risk)
        if level is not None
    }
    if "HIGH" in risk_levels:
        marker_color = "red"
    elif "MEDIUM" in risk_levels:
        marker_color = "orange"
    elif "LOW" in risk_levels:
        marker_color = "green"
    else:
        marker_color = "blue"

    district_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.Marker(
        location=[latitude, longitude],
        popup=folium.Popup("<br>".join(popup_parts), max_width=300),
        tooltip=district.title(),
        icon=folium.Icon(color=marker_color, icon="info-sign"),
    ).add_to(district_map)

    st_folium(
        district_map,
        width=700,
        height=map_height,
        returned_objects=[],
        key=map_key,
    )
