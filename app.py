import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("parkinsons_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set up the page
st.set_page_config(page_title="Parkinson's Predictor", page_icon="🧠", layout="centered")
st.markdown("<h1 style='text-align: center;'>🧠 Parkinson's Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict the likelihood of Parkinson's disease based on key voice features.</p>", unsafe_allow_html=True)
st.markdown("---")

# Top 7 most important features
features = {
    "MDVP:Fo(Hz)": "Average vocal frequency",
    "MDVP:Jitter(%)": "Pitch variation (jitter)",
    "MDVP:Shimmer": "Amplitude variation (shimmer)",
    "HNR": "Harmonics-to-noise ratio",
    "RPDE": "Recurrence period density entropy",
    "DFA": "Fractal scaling exponent",
    "PPE": "Nonlinear pitch variation"
}

# Layout for inputs
col1, col2 = st.columns(2)
inputs = []

for i, (label, help_text) in enumerate(features.items()):
    value = (col1 if i % 2 == 0 else col2).number_input(label, help=help_text, format="%.5f")
    inputs.append(value)

# Prediction
if st.button("🔍 Predict"):
    input_array = np.array([inputs])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][prediction]

    st.markdown("---")
    if prediction == 1:
        st.error(f"🔴 Parkinson's Detected! Confidence: {prob * 100:.2f}%")
        st.info("Please consult a medical expert for further testing.")
    else:
        st.success(f"🟢 No Parkinson's Detected! Confidence: {prob * 100:.2f}%")
        st.balloons()
