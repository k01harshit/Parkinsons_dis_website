import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("parkinsons_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Parkinson's Predictor", page_icon="🧠", layout="centered")
st.markdown("<h1 style='text-align: center;'>🧠 Parkinson's Disease Predictor</h1>", unsafe_allow_html=True)
st.write("Enter the following key voice features to predict the likelihood of Parkinson's Disease.")

# Selected top features
features = {
    "MDVP:Fo(Hz)": "Average vocal fundamental frequency",
    "MDVP:Jitter(%)": "Variation in pitch (jitter)",
    "MDVP:Shimmer": "Amplitude variation (shimmer)",
    "HNR": "Harmonics-to-noise ratio",
    "RPDE": "Signal complexity (RPDE)",
    "DFA": "Signal fractal scaling exponent (DFA)",
    "PPE": "Non-linear variation in pitch (PPE)"
}

# Layout in two columns
col1, col2 = st.columns(2)
inputs = []

for i, (feature, desc) in enumerate(features.items()):
    if i % 2 == 0:
        value = col1.number_input(f"{feature}", help=desc, format="%.5f")
    else:
        value = col2.number_input(f"{feature}", help=desc, format="%.5f")
    inputs.append(value)

# Predict
if st.button("🔍 Predict"):
    input_array = np.array([inputs])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    st.markdown("---")
    if prediction == 1:
        st.error(f"🔴 **Parkinson's Detected**

Confidence: **{probability * 100:.2f}%**")
        st.info("Please consult a medical professional for further diagnosis and treatment options.")
    else:
        st.success(f"🟢 **No Parkinson's Detected**)

Confidence: **{probability * 100:.2f}%**")
        st.balloons()
