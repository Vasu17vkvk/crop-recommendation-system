import streamlit as st
import pandas as pd
import joblib

# ======================
# LOAD MODEL
# ======================

model = joblib.load("india_crop_model_v3.pkl")
encoder = joblib.load("india_crop_encoder_v3.pkl")

st.title("🌾 Crop Recommendation System")

st.write("Enter feature values")

# ======================
# INPUTS
# ======================

NDVI = st.number_input("NDVI", value=0.50)
EVI = st.number_input("EVI", value=0.30)
NDWI = st.number_input("NDWI", value=0.10)

Elevation = st.number_input("Elevation", value=200.0)
Rainfall = st.number_input("Rainfall", value=800.0)

Nitrogen_High = st.number_input("Nitrogen High", value=20.0)
Nitrogen_Medium = st.number_input("Nitrogen Medium", value=60.0)
Nitrogen_Low = st.number_input("Nitrogen Low", value=20.0)

Phosphorous_High = st.number_input("Phosphorous High", value=20.0)
Phosphorous_Medium = st.number_input("Phosphorous Medium", value=60.0)
Phosphorous_Low = st.number_input("Phosphorous Low", value=20.0)

Potassium_High = st.number_input("Potassium High", value=20.0)
Potassium_Medium = st.number_input("Potassium Medium", value=60.0)
Potassium_Low = st.number_input("Potassium Low", value=20.0)

pH_Acidic = st.number_input("pH Acidic", value=5.0)
pH_Neutral = st.number_input("pH Neutral", value=90.0)
pH_Alkaline = st.number_input("pH Alkaline", value=5.0)

# ======================
# PREDICT
# ======================

if st.button("Predict Crop"):

    data = pd.DataFrame([{
        "NDVI": NDVI,
        "EVI": EVI,
        "NDWI": NDWI,

        "Elevation": Elevation,
        "Rainfall": Rainfall,

        "Nitrogen_High": Nitrogen_High,
        "Nitrogen_Medium": Nitrogen_Medium,
        "Nitrogen_Low": Nitrogen_Low,

        "Phosphorous_High": Phosphorous_High,
        "Phosphorous_Medium": Phosphorous_Medium,
        "Phosphorous_Low": Phosphorous_Low,

        "Potassium_High": Potassium_High,
        "Potassium_Medium": Potassium_Medium,
        "Potassium_Low": Potassium_Low,

        "pH_Acidic": pH_Acidic,
        "pH_Neutral": pH_Neutral,
        "pH_Alkaline": pH_Alkaline
    }])

prediction = model.predict(data)

crop = encoder.inverse_transform(prediction)

probabilities = model.predict_proba(data)[0]

top3_idx = probabilities.argsort()[-3:][::-1]

st.success(
    f"🌾 Recommended Crop: {crop[0]}"
)

st.subheader("Top 3 Crops")

for idx in top3_idx:

    crop_name = encoder.inverse_transform([idx])[0]

    confidence = probabilities[idx] * 100

    st.write(
        f"{crop_name} : {confidence:.2f}%"
    )
