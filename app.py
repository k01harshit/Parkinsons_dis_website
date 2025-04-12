import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    df = pd.read_csv(url)
    return df

df = load_data()

# Selected features and target
selected_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer',
                     'NHR', 'HNR', 'RPDE', 'DFA']
X = df[selected_features]
y = df['status']

# Scale and split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Title
st.title("🧠 Parkinson's Disease Prediction (No .pkl files)")
st.markdown("Enter your voice measurement parameters:")

# Input fields
fo = st.number_input("1. MDVP:Fo(Hz) – Average vocal fundamental frequency", 50.0, 300.0, 150.0)
jitter = st.number_input("2. MDVP:Jitter(%) – Frequency variation", 0.0, 1.0, 0.005)
shimmer = st.number_input("3. MDVP:Shimmer – Amplitude variation", 0.0, 1.0, 0.03)
nhr = st.number_input("4. NHR – Noise-to-harmonics ratio", 0.0, 1.0, 0.02)
hnr = st.number_input("5. HNR – Harmonics-to-noise ratio", 0.0, 50.0, 20.0)
rpde = st.number_input("6. RPDE – Recurrence Period Density Entropy", 0.0, 1.0, 0.5)
dfa = st.number_input("7. DFA – Detrended Fluctuation Analysis", 0.0, 2.0, 0.7)

# Predict
if st.button("🔍 Predict"):
    user_input = np.array([[fo, jitter, shimmer, nhr, hnr, rpde, dfa]])
    user_scaled = scaler.transform(user_input)
    prediction = model.predict(user_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Likely Parkinson's Disease")
    else:
        st.success("✅ Unlikely to Have Parkinson's Disease")
