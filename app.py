import streamlit as st
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Safe loading with fallback
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"❌ File not found: {path}")
        st.stop()

def load_scaler(path, feature_count):
    if os.path.exists(path):
        scaler = joblib.load(path)
        if hasattr(scaler, "transform"):
            return scaler
        else:
            st.warning("⚠️ Invalid scaler detected. Using a fake one for testing.")
    else:
        st.warning("⚠️ Scaler file not found. Using a fake scaler for testing.")
    
    # Create a dummy scaler (does nothing)
    fake_scaler = StandardScaler()
    fake_scaler.mean_ = np.zeros(feature_count)
    fake_scaler.scale_ = np.ones(feature_count)
    fake_scaler.var_ = np.ones(feature_count)
    return fake_scaler

# Load model and scaler
model = load_model("parkinsons_model.pkl")

# Define features first
features = {
    "MDVP:Fo(Hz)": "Average vocal frequency",
    "MDVP:Jitter(%)": "Pitch variation (jitter)",
    "MDVP:Shimmer": "Amplitude variation (shimmer)",
    "HNR": "Harmonics-to-noise ratio",
    "RPDE": "Recurrence period density entropy",
    "DFA": "Fractal scaling exponent",
    "PPE": "Nonlinear pitch variation"
}

scaler = load_scaler("scaler.pkl", feature_count=len(features))

# Set up the page
st.set_page_config(page_title="Parkinson's Predictor", page_icon="🧠", layout="centered")
st.markdown("<h1 style='text-align: center;'>🧠 Parkinson's Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict the likelihood of Parkinson's disease based on key voice features.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input layout
col1, col2 = st.columns(2)
inputs = []

for i, (label, help_text) in enumerate(features.items()):
    value = (col1 if i % 2 == 0 else col2).number_input(
        label, help=help_text, format="%.5f", value=0.0
    )
    inputs.append(value)

# Prediction
if st.button("🔍 Predict"):
    try:
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
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")


