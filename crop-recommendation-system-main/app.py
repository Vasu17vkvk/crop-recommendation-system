import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import ee
import json

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Crop Recommendation System",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("india_crop_model_v3.pkl")
encoder = joblib.load("india_crop_encoder_v3.pkl")

soil_df = pd.read_csv("India_Crop_Features.csv")

# =====================================================
# EARTH ENGINE INIT
# =====================================================

try:
    service_account_info = dict(st.secrets["gcp_service_account"])
    credentials = ee.ServiceAccountCredentials(
        service_account_info["client_email"],
        key_data=json.dumps(service_account_info)
    )
except:
    credentials = ee.ServiceAccountCredentials(
        None,
        "service-account.json"
    )

ee.Initialize(credentials)

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def get_features(lat, lon):
    point = ee.Geometry.Point([lon, lat])

    image = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate("2024-01-01", "2024-12-31")
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
    )

    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

    evi = image.expression(
        "2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
        {
            "NIR": image.select("B8"),
            "RED": image.select("B4"),
            "BLUE": image.select("B2")
        }
    ).rename("EVI")

    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")

    elevation = ee.Image("USGS/SRTMGL1_003").rename("Elevation")

    rainfall = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate("2024-01-01", "2024-12-31")
        .sum()
        .rename("Rainfall")
    )

    values = (
        ee.Image.cat([ndvi, evi, ndwi, elevation, rainfall])
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30,
            maxPixels=1e9
        )
    )

    return values.getInfo()

# =====================================================
# NEAREST SOIL DATA
# =====================================================

def get_nearest_soil(lat, lon):
    temp = soil_df.copy()

    temp["distance"] = (
        (temp["Latitude"] - lat) ** 2 +
        (temp["Longitude"] - lon) ** 2
    )

    nearest5 = temp.nsmallest(5, "distance")

    return {
        "Nitrogen_High": nearest5["Nitrogen_High"].mean(),
        "Nitrogen_Medium": nearest5["Nitrogen_Medium"].mean(),
        "Nitrogen_Low": nearest5["Nitrogen_Low"].mean(),
        "Phosphorous_High": nearest5["Phosphorous_High"].mean(),
        "Phosphorous_Medium": nearest5["Phosphorous_Medium"].mean(),
        "Phosphorous_Low": nearest5["Phosphorous_Low"].mean(),
        "Potassium_High": nearest5["Potassium_High"].mean(),
        "Potassium_Medium": nearest5["Potassium_Medium"].mean(),
        "Potassium_Low": nearest5["Potassium_Low"].mean(),
        "pH_Acidic": nearest5["pH_Acidic"].mean(),
        "pH_Neutral": nearest5["pH_Neutral"].mean(),
        "pH_Alkaline": nearest5["pH_Alkaline"].mean(),
        "Region": nearest5.iloc[0]["Region"],
        "Address": nearest5.iloc[0]["Address"]
    }

# =====================================================
# TITLE
# =====================================================

st.title("🌾 AI Crop Recommendation System")

st.write("Click anywhere on the map to automatically fetch satellite data and predict the best crop.")

# =====================================================
# MAP
# =====================================================

m = folium.Map(
    location=[23.5, 80],
    zoom_start=5
)

map_data = st_folium(
    m,
    width=1000,
    height=500
)

# =====================================================
# USER CLICK
# =====================================================

if map_data.get("last_clicked"):

    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    st.success(f"Latitude : {lat}")
    st.success(f"Longitude : {lon}")

    with st.spinner("Fetching satellite features..."):
        try:
            features = get_features(lat, lon)
            soil = get_nearest_soil(lat, lon)

            ndvi = float(features.get("NDVI", 0))
            evi = float(features.get("EVI", 0))
            ndwi = float(features.get("NDWI", 0))
            elevation = float(features.get("Elevation", 0))
            rainfall = float(features.get("Rainfall", 0))

            # =====================================================
            # SATELLITE FEATURES
            # =====================================================

            st.subheader("🛰️ Satellite Features")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("NDVI", round(ndvi, 4))
            with col2:
                st.metric("EVI", round(evi, 4))
            with col3:
                st.metric("NDWI", round(ndwi, 4))

            col4, col5 = st.columns(2)
            with col4:
                st.metric("Elevation (m)", round(elevation, 2))
            with col5:
                st.metric("Rainfall (mm)", round(rainfall, 2))

            # ===================================
            # MOISTURE STRESS DETECTION
            # ===================================

            st.subheader("💧 Moisture Stress Analysis")

            if ndwi > 0.2:
                stress_level = "🟢 Low Stress"
                stress_msg = "Crop moisture condition is healthy."
            elif ndwi > 0:
                stress_level = "🟡 Moderate Stress"
                stress_msg = "Crop may require monitoring."
            else:
                stress_level = "🔴 High Stress"
                stress_msg = "Possible moisture stress detected."

            st.success(stress_level)
            st.info(stress_msg)

            # ===================================
            # IRRIGATION ADVISORY
            # ===================================

            st.subheader("🚿 Irrigation Advisory")

            if ndwi < 0 and rainfall < 800:
                st.error("🚨 Irrigation Required Immediately")
                st.write("Low vegetation water content and limited rainfall detected.")
            elif ndwi < 0 and rainfall >= 800:
                st.warning("⚠️ Monitor Field Conditions")
                st.write("Moisture stress detected, but rainfall levels are relatively high.")
            elif ndwi >= 0 and rainfall < 500:
                st.warning("⚠️ Light Irrigation Recommended")
                st.write("Current moisture is acceptable, but rainfall is low.")
            else:
                st.success("✅ No Immediate Irrigation Required")
                st.write("Moisture and rainfall conditions appear adequate.")

            # =====================================================
            # EXISTING SOIL SLIDERS (TEMPORARY)
            # =====================================================

            st.subheader("🧪 Soil Parameters")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Nitrogen High", round(soil["Nitrogen_High"], 2))
                st.metric("Nitrogen Medium", round(soil["Nitrogen_Medium"], 2))
                st.metric("Nitrogen Low", round(soil["Nitrogen_Low"], 2))

            with col2:
                st.metric("Phosphorous High", round(soil["Phosphorous_High"], 2))
                st.metric("Phosphorous Medium", round(soil["Phosphorous_Medium"], 2))
                st.metric("Phosphorous Low", round(soil["Phosphorous_Low"], 2))

            with col3:
                st.metric("Potassium High", round(soil["Potassium_High"], 2))
                st.metric("Potassium Medium", round(soil["Potassium_Medium"], 2))
                st.metric("Potassium Low", round(soil["Potassium_Low"], 2))

            st.subheader("⚗️ Soil pH")

            col4, col5, col6 = st.columns(3)

            with col4:
                st.metric("Acidic", round(soil["pH_Acidic"], 2))
            with col5:
                st.metric("Neutral", round(soil["pH_Neutral"], 2))
            with col6:
                st.metric("Alkaline", round(soil["pH_Alkaline"], 2))

            # =====================================================
            # USER INPUTS FOR PREDICTION
            # =====================================================
            
            st.subheader("🌱 Growth Stage Selection")
            growth_stage = st.selectbox(
                "Select current growth stage to refine advisory:",
                ["Seedling", "Vegetative", "Flowering", "Maturity"]
            )

            # =====================================================
            # PREDICTION
            # =====================================================

            if st.button("🌾 Predict Crop"):

                data = pd.DataFrame([{
                    "NDVI": ndvi,
                    "EVI": evi,
                    "NDWI": ndwi,
                    "Elevation": elevation,
                    "Rainfall": rainfall,
                    "Nitrogen_High": soil["Nitrogen_High"],
                    "Nitrogen_Medium": soil["Nitrogen_Medium"],
                    "Nitrogen_Low": soil["Nitrogen_Low"],
                    "Phosphorous_High": soil["Phosphorous_High"],
                    "Phosphorous_Medium": soil["Phosphorous_Medium"],
                    "Phosphorous_Low": soil["Phosphorous_Low"],
                    "Potassium_High": soil["Potassium_High"],
                    "Potassium_Medium": soil["Potassium_Medium"],
                    "Potassium_Low": soil["Potassium_Low"],
                    "pH_Acidic": soil["pH_Acidic"],
                    "pH_Neutral": soil["pH_Neutral"],
                    "pH_Alkaline": soil["pH_Alkaline"]
                }])

                prediction = model.predict(data)
                crop = encoder.inverse_transform(prediction)
                probabilities = model.predict_proba(data)[0]
                
                top3_idx = probabilities.argsort()[-3:][::-1]

                st.success(f"🌾 Recommended Crop: {crop[0]}")

                crop_name = crop[0].lower()
                
                st.subheader("🚿 Stage-wise Irrigation Advisory")

                crop_name = crop[0].lower()

                if crop_name == "paddy":

                    if growth_stage == "Seedling":
                        st.info("Maintain standing water and avoid soil drying.")

                    elif growth_stage == "Vegetative":
                        st.info("Maintain shallow water depth for active growth.")

                    elif growth_stage == "Flowering":
                        st.warning("Critical stage. Water shortage can severely reduce yield.")

                    elif growth_stage == "Maturity":
                        st.success("Reduce irrigation before harvest.")

                elif crop_name == "wheat":

                    if growth_stage == "Seedling":
                        st.info("Keep soil moist for establishment.")

                    elif growth_stage == "Vegetative":
                        st.info("Moderate irrigation supports tillering.")

                    elif growth_stage == "Flowering":
                        st.warning("Avoid moisture stress during flowering.")

                    elif growth_stage == "Maturity":
                        st.success("Limit irrigation near harvest.")

                elif crop_name == "maize":

                    if growth_stage == "Seedling":
                        st.info("Ensure adequate moisture for root development.")

                    elif growth_stage == "Vegetative":
                        st.info("Moderate irrigation recommended.")

                    elif growth_stage == "Flowering":
                        st.warning("Water stress at flowering can reduce grain yield.")

                    elif growth_stage == "Maturity":
                        st.success("Reduce irrigation as crop approaches harvest.")

                else:

                    st.info(
                        f"General irrigation guidance for {crop[0]} at {growth_stage} stage."
                    )

    
                st.subheader("🏆 Top 3 Crop Recommendations")

                medals = ["🥇", "🥈", "🥉"]

                for rank, idx in enumerate(top3_idx):
                    crop_name = encoder.inverse_transform([idx])[0]
                    confidence = probabilities[idx] * 100
                    st.write(f"{medals[rank]} {crop_name} : {confidence:.2f}%")
                    st.progress(float(confidence / 100))

        except Exception as e:
            st.error(f"Earth Engine Error: {str(e)}")
