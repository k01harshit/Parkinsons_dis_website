import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Set Streamlit page config
st.set_page_config(page_title="Parkinson's Disease Prediction", layout="centered")

# App title
st.title("🧠 Parkinson's Disease Prediction App")
st.markdown("Provide the top 7 voice features to predict Parkinson's Disease.")

# Load dataset locally
@st.cache_data
def load_data():
    df = pd.read_csv("parkinsons.csv")
    return df

# Load data
df = load_data()

# Select top 7 features
features = ['MDVP:Fo(Hz)',       # Average vocal fundamental frequency
            'MDVP:Jitter(%)',    # Variability in frequency
            'MDVP:Shimmer',      # Variability in amplitude
            'NHR',               # Noise-to-harmonics ratio
            'HNR',               # Harmonics-to-noise ratio
            'RPDE',              # Signal complexity
            'DFA']              # Nonlinear dynamical complexity

X = df[features]
y = df['status']  # 1 = Parkinson's, 0 = Healthy

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# --- Streamlit Inputs ---
st.subheader("🔍 Enter Voice Feature Values")

input_data = []
for feature in features:
    value = st.number_input(f"{feature}", format="%.5f")
    input_data.append(value)

# Predict button
if st.button("🧪 Predict"):
    input_np = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_np)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.error(f"🚨 Prediction: Parkinson's Disease Detected (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Prediction: No Parkinson's Detected (Confidence: {probability:.2f})")
