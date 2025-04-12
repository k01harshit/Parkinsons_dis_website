import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("parkinsons_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Parkinson's Prediction", layout="centered")
st.title("🧠 Parkinson's Disease Prediction App")
st.write("Enter the patient’s voice measurements below to predict the likelihood of Parkinson's disease.")

# Define feature labels
features = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
    "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

# Split inputs into two columns for layout
col1, col2 = st.columns(2)
inputs = []

for i, feature in enumerate(features):
    if i % 2 == 0:
        value = col1.number_input(f"{feature}", format="%.5f")
    else:
        value = col2.number_input(f"{feature}", format="%.5f")
    inputs.append(value)

# Predict button
if st.button("Predict"):
    input_array = np.array([inputs])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.error(f"🔴 Parkinson's Detected with {probability * 100:.2f}% confidence.")
    else:
        st.success(f"🟢 No Parkinson's Detected with {probability * 100:.2f}% confidence.")