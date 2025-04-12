import streamlit as st
import numpy as np
import joblib

# Load model and scaler
@st.cache_resource
def load_model_scaler():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_scaler()

# Full-form features to show nicely
feature_labels = {
    "MDVP:Fo(Hz)": "Average vocal fundamental frequency (Hz)",
    "MDVP:Jitter(%)": "Variation in vocal frequency (Jitter %)",
    "MDVP:Shimmer": "Variation in vocal amplitude (Shimmer)",
    "NHR": "Noise-to-Harmonics Ratio",
    "HNR": "Harmonics-to-Noise Ratio",
    "RPDE": "Recurrence Period Density Entropy",
    "DFA": "Detrended Fluctuation Analysis"
}

st.title("🧠 Parkinson's Disease Prediction App")

st.markdown(
    """
    Enter the following **voice measurements** to check whether the person is likely to have **Parkinson's disease**.
    """
)

# Input form
with st.form("input_form"):
    inputs = {}
    for key, label in feature_labels.items():
        inputs[key] = st.number_input(f"{label} ({key})", step=0.01, format="%.4f")

    submitted = st.form_submit_button("Predict")

if submitted:
    input_values = np.array(list(inputs.values())).reshape(1, -1)

    # Scale and Predict
    try:
        input_scaled = scaler.transform(input_values)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"🧪 The person **may have Parkinson's Disease**. (Confidence: {probability:.2%})")
        else:
            st.success(f"✅ The person is **unlikely to have Parkinson's Disease**. (Confidence: {1 - probability:.2%})")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
