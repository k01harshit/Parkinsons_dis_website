import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Streamlit page config
st.set_page_config(page_title="Parkinson's Disease Predictor", layout="centered")

# Title
st.title("🧠 Parkinson's Disease Prediction App")

# Description
st.markdown("Provide the values for the following **7 voice features** to predict the likelihood of Parkinson’s Disease.")

# Try loading model and scaler
try:
    model = joblib.load("model.pkl")  # Replace with your model path
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    model = None
    scaler = None
    st.warning("⚠️ Model or scaler not loaded. Prediction won't work without them.")

# Define feature names and full forms
features = {
    'MDVP:Fo(Hz)': "Average vocal fundamental frequency",
    'MDVP:Jitter(%)': "Variation in fundamental frequency",
    'MDVP:Shimmer': "Variation in amplitude",
    'NHR': "Noise-to-Harmonics Ratio",
    'HNR': "Harmonics-to-Noise Ratio",
    'RPDE': "Recurrence Period Density Entropy",
    'DFA': "Detrended Fluctuation Analysis"
}

# Input form
with st.form("parkinsons_form"):
    user_input = []
    for key, desc in features.items():
        val = st.number_input(f"{key} ({desc})", min_value=0.0, step=0.01, format="%.4f")
        user_input.append(val)
    
    submit = st.form_submit_button("🔍 Predict")

# Prediction
if submit:
    if model is None or scaler is None:
        st.error("❌ Model or Scaler not available. Please upload them.")
    else:
        input_array = np.array(user_input).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.error("🧬 The model predicts: **Parkinson’s Disease Detected.**")
        else:
            st.success("✅ The model predicts: **No Parkinson’s Disease Detected.**")

# Optionally: Upload CSV and show structure
st.sidebar.header("🗂️ Upload CSV (optional)")
uploaded_file = st.sidebar.file_uploader("Upload a Parkinson's dataset CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Clean up column names
    st.sidebar.write("📊 CSV Columns:", df.columns.tolist())
