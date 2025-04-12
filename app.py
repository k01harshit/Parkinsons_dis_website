import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

st.set_page_config(page_title="Parkinson's Disease Prediction", layout="centered")
st.title("🧠 Parkinson's Disease Prediction App")

@st.cache_data
def load_data():
    if os.path.exists("parkinsons.csv"):
        return pd.read_csv("parkinsons.csv")
    else:
        return None

df = load_data()

# Upload fallback
if df is None:
    uploaded_file = st.file_uploader("Upload the 'parkinsons.csv' file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.to_csv("parkinsons.csv", index=False)  # Save for future use
        st.success("✅ File uploaded successfully!")
    else:
        st.warning("⚠️ Please upload the dataset to proceed.")
        st.stop()

# Continue if data is available
features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA']
X = df[features]
y = df['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

st.subheader("🔍 Enter Voice Feature Values")
input_data = [st.number_input(f"{f}", format="%.5f") for f in features]

if st.button("🧪 Predict"):
    input_np = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_np)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.error(f"🚨 Parkinson's Detected! (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ No Parkinson's Detected. (Confidence: {prob:.2f})")
