import streamlit as st
import numpy as np
import joblib
import os

# ----------- Setup: Load model and scaler -----------
if not os.path.exists("best_model.pkl") or not os.path.exists("scaler.pkl"):
    st.error("❌ Required files not found. Ensure 'best_model.pkl' and 'scaler.pkl' are in the app directory.")
    st.stop()

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------- Page Configuration -----------
st.set_page_config(page_title="🧠 Parkinson's Prediction", layout="centered")

# ----------- Mode Toggle -----------
st.markdown(
    """
    <style>
        .dark-mode { background-color: #121212; color: #FFFFFF; }
        .light-mode { background-color: #FFFFFF; color: #000000; }
        .centered { text-align: center; }
        .input-label { font-weight: 600; font-size: 16px; margin-bottom: 5px; }
        .main-block { background-color: rgba(255,255,255,0.07); padding: 2rem; border-radius: 15px; box-shadow: 0 0 10px rgba(0,0,0,0.2); }
    </style>
    """,
    unsafe_allow_html=True
)

dark_mode = st.toggle("🌗 Toggle Dark Mode", value=True)
page_mode = "dark-mode" if dark_mode else "light-mode"

st.markdown(f'<div class="{page_mode}">', unsafe_allow_html=True)

# ----------- Header -----------
st.markdown(f"""
<div class="centered">
    <h1>🧠 Parkinson's Disease Predictor</h1>
    <h4>Enter your voice and neuromotor biometrics to check Parkinson's risk</h4>
</div>
""", unsafe_allow_html=True)

# ----------- Feature Metadata -----------
feature_labels = {
    'MDVP:Fo(Hz)': 'Average Vocal Fundamental Frequency',
    'MDVP:Fhi(Hz)': 'Maximum Vocal Fundamental Frequency',
    'MDVP:Flo(Hz)': 'Minimum Vocal Fundamental Frequency',
    'MDVP:Jitter(%)': 'Frequency Variation (Jitter)',
    'MDVP:Jitter(Abs)': 'Absolute Jitter',
    'MDVP:RAP': 'Relative Amplitude Perturbation',
    'MDVP:PPQ': 'Pitch Perturbation Quotient',
    'Jitter:DDP': 'Three-point Period Perturbation Quotient',
    'MDVP:Shimmer': 'Amplitude Variation (Shimmer)',
    'MDVP:Shimmer(dB)': 'Shimmer in Decibels',
    'Shimmer:APQ3': 'Amplitude Perturbation Quotient 3',
    'Shimmer:APQ5': 'Amplitude Perturbation Quotient 5',
    'MDVP:APQ': 'Average Amplitude Perturbation Quotient',
    'Shimmer:DDA': 'Difference of Differences of Amplitude',
    'NHR': 'Noise-to-Harmonics Ratio',
    'HNR': 'Harmonics-to-Noise Ratio',
    'RPDE': 'Recurrence Period Density Entropy',
    'DFA': 'Detrended Fluctuation Analysis',
    'spread1': 'Non-linear measure of F0 variation',
    'spread2': 'Non-linear measure of amplitude variation',
    'D2': 'Correlation Dimension',
    'PPE': 'Pitch Period Entropy'
}

required_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA']
optional_features = [f for f in feature_labels if f not in required_features]

user_inputs = {}

# ----------- Input Form -----------
with st.form("prediction_form"):
    st.markdown('<div class="main-block">', unsafe_allow_html=True)
    st.subheader("🔹 Required Inputs")
    for feat in required_features:
        user_inputs[feat] = st.number_input(
            f"{feature_labels[feat]} ({feat})", format="%.5f", key=feat
        )

    st.subheader("🔸 Optional Inputs (leave blank if unknown)")
    for feat in optional_features:
        val = st.text_input(f"{feature_labels[feat]} ({feat})", key=feat)
        try:
            user_inputs[feat] = float(val) if val else 0.0
        except ValueError:
            st.warning(f"⚠️ Please enter a valid number for {feat}")
            st.stop()
    st.markdown('</div>', unsafe_allow_html=True)

    submit = st.form_submit_button("🧪 Predict Now")

# ----------- Prediction Logic -----------
if submit:
    try:
        input_array = np.array([user_inputs[feat] for feat in feature_labels])
        input_scaled = scaler.transform([input_array])
        prediction = model.predict(input_scaled)[0]

        st.markdown("---")
        if prediction == 1:
            st.error("⚠️ The model predicts that the patient **may have Parkinson's Disease**.")
        else:
            st.success("✅ The model predicts that the patient **does not have Parkinson's Disease**.")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

st.markdown("</div>", unsafe_allow_html=True)
