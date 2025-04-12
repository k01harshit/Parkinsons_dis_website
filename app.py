import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Check if required files exist
if not os.path.exists("best_model.pkl") or not os.path.exists("scaler.pkl"):
    st.error("❌ Required model or preprocessing files not found. Please ensure 'best_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit page config
st.set_page_config(page_title="Parkinson's Disease Predictor", layout="centered")

# Toggle theme
dark_mode = st.toggle("🌙 Dark Mode", value=True)

# Apply CSS styling
st.markdown(
    f"""
    <style>
        body {{
            background-color: {'#0e1117' if dark_mode else '#ffffff'};
            color: {'#f5f6fa' if dark_mode else '#000000'};
        }}
        .stTextInput > div > div > input {{
            border-radius: 10px;
            padding: 8px;
        }}
        .stButton>button {{
            background-color: {'#4CAF50' if dark_mode else '#2c3e50'};
            color: white;
            border: none;
            border-radius: 12px;
            padding: 10px 20px;
            font-size: 16px;
            margin-top: 10px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# App title and instructions
st.title("🧠 Parkinson's Disease Prediction App")
st.markdown("Enter your medical voice parameters below to check the likelihood of having Parkinson's Disease.")

# Full feature names
feature_names = {
    'MDVP:Fo(Hz)': 'Average Vocal Fundamental Frequency',
    'MDVP:Jitter(%)': 'Frequency Variation (Jitter)',
    'MDVP:Shimmer': 'Amplitude Variation (Shimmer)',
    'NHR': 'Noise-to-Harmonics Ratio',
    'HNR': 'Harmonics-to-Noise Ratio',
    'RPDE': 'Recurrence Period Density Entropy',
    'DFA': 'Detrended Fluctuation Analysis',
    'spread1': 'Non-linear measure of fundamental frequency variation',
    'spread2': 'Non-linear measure of amplitude variation',
    'PPE': 'Pitch Period Entropy',
    'MDVP:APQ': 'Amplitude Perturbation Quotient',
    'MDVP:PPQ': 'Pitch Perturbation Quotient',
    'MDVP:RAP': 'Relative Amplitude Perturbation',
    'D2': 'Correlation dimension',
    'MDVP:Fhi(Hz)': 'Maximum Vocal Fundamental Frequency',
    'MDVP:Flo(Hz)': 'Minimum Vocal Fundamental Frequency'
}

# Top 7 required features
required_features = [
    'MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer',
    'NHR', 'HNR', 'RPDE', 'DFA'
]

# Optional features
optional_features = [f for f in feature_names if f not in required_features]

user_data = {}

st.subheader("🔹 Required Inputs (Mandatory)")
for feat in required_features:
    val = st.number_input(f"{feature_names[feat]} ({feat})", format="%.5f", key=feat)
    user_data[feat] = val

st.subheader("🔸 Optional Inputs")
for feat in optional_features:
    val = st.number_input(f"{feature_names[feat]} ({feat})", format="%.5f", key=feat)
    user_data[feat] = val

# Prediction
if st.button("🧪 Predict"):
    try:
        input_array = np.array([user_data[feat] for feat in feature_names])
        input_scaled = scaler.transform([input_array])
        prediction = model.predict(input_scaled)[0]

        st.success("✅ Prediction complete!")
        if prediction == 1:
            st.error("⚠️ The model predicts that the patient **may have Parkinson's Disease**.")
        else:
            st.success("🎉 The model predicts that the patient **does not have Parkinson's Disease**.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
