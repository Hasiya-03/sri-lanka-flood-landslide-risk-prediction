
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium
from pathlib import Path

# Approximate district-centre coordinates for map markers
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
    "vavuniya": (8.75, 80.50)
}


def show_district_map(
    selected_district,
    flood_risk=None,
    landslide_risk=None
):
    # Load existing project terrain data
    data_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "district_terrain.csv"
    )

    terrain_data = pd.read_csv(data_path)
    terrain_data["District"] = (
        terrain_data["District"].str.lower().str.strip()
    )

    district = selected_district.lower().strip()

    if district not in DISTRICT_COORDINATES:
        st.warning("Location not available for this district.")
        return

    latitude, longitude = DISTRICT_COORDINATES[district]

    # Create the interactive map
    district_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=10,
        tiles="OpenStreetMap"
    )

    district_row = terrain_data[
        terrain_data["District"] == district
    ]

    if not district_row.empty:
        row = district_row.iloc[0]

        popup_text = (
            f"<b>District:</b> {district.title()}<br>"
            f"<b>Mean Elevation:</b> "
            f"{row['Mean_Elevation_m']:.2f} m<br>"
            f"<b>Mean Slope:</b> "
            f"{row['Mean_Slope_deg']:.2f}°"
        )
    else:
        popup_text = district.title()

            # Include AI prediction results when available
    if flood_risk is not None:
        popup_text += (
            f"<br><b>Flood Risk:</b> {flood_risk}"
        )

    if landslide_risk is not None:
        popup_text += (
            f"<br><b>Landslide Risk:</b> {landslide_risk}"
        )


    risk_levels = [
        str(flood_risk).upper() if flood_risk is not None else None,
        str(landslide_risk).upper() if landslide_risk is not None else None
    ]
    if "HIGH" in risk_levels:
        marker_color = "red"
    elif "MEDIUM" in risk_levels:
        marker_color = "orange"
    elif "LOW" in risk_levels:
        marker_color = "green"
    else:
        marker_color = "blue"

    folium.Marker(
        location=[latitude, longitude],
        popup=folium.Popup(popup_text, max_width=300),
        tooltip=district.title(),
        icon=folium.Icon(
            color=marker_color,
            icon="info-sign"
        )
    ).add_to(district_map)

    st_folium(
        district_map,
        width="stretch",
        height=450,
        returned_objects=[]
    )
