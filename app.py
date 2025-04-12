import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load Data
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    df = pd.read_csv(url)
    return df

df = load_data()

# 2. Preprocess
selected_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA']
X = df[selected_features]
y = df['status']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split (not used directly, but could be extended later)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. App UI
st.set_page_config(page_title="Parkinson's Predictor", layout="centered", page_icon="🧠")
st.markdown("<h1 style='text-align: center; color: #6c63ff;'>🧠 Parkinson's Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("Enter voice measurement data below or upload a CSV for batch predictions.")

# 4. User Inputs
st.subheader("🔬 Manual Input")

cols = st.columns(2)
fo = cols[0].number_input("1. MDVP:Fo(Hz)", min_value=50.0, max_value=300.0, value=150.0, help="Average vocal fundamental frequency")
jitter = cols[1].number_input("2. MDVP:Jitter(%)", 0.0, 1.0, 0.005, help="Frequency variation in voice")
shimmer = cols[0].number_input("3. MDVP:Shimmer", 0.0, 1.0, 0.03, help="Amplitude variation")
nhr = cols[1].number_input("4. NHR", 0.0, 1.0, 0.02, help="Noise-to-harmonics ratio")
hnr = cols[0].number_input("5. HNR", 0.0, 50.0, 20.0, help="Harmonics-to-noise ratio")
rpde = cols[1].number_input("6. RPDE", 0.0, 1.0, 0.5, help="Nonlinear dynamics measure")
dfa = cols[0].number_input("7. DFA", 0.0, 2.0, 0.7, help="Signal fractal scaling exponent")

# 5. Predict
if st.button("🔍 Predict Now"):
    input_data = np.array([[fo, jitter, shimmer, nhr, hnr, rpde, dfa]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1] * 100

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **Likely Parkinson's Disease**\n\n🧪 Confidence: {proba:.2f}%")
    else:
        st.success(f"✅ **Unlikely to Have Parkinson's**\n\n🧪 Confidence: {100-proba:.2f}%")

# 6. Upload CSV
st.markdown("---")
st.subheader("📁 Batch Prediction via CSV")
uploaded_file = st.file_uploader("Upload CSV file with 7 feature columns:", type=['csv'])

if uploaded_file:
    try:
        user_df = pd.read_csv(uploaded_file)
        if set(selected_features).issubset(user_df.columns):
            X_user = user_df[selected_features]
            X_user_scaled = scaler.transform(X_user)
            predictions = model.predict(X_user_scaled)
            probabilities = model.predict_proba(X_user_scaled)[:, 1]

            result_df = user_df.copy()
            result_df['Prediction'] = ["Parkinson's" if p == 1 else "Healthy" for p in predictions]
            result_df['Confidence (%)'] = (probabilities * 100).round(2)

            st.success("✅ Prediction completed!")
            st.dataframe(result_df)

            # Download button
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results as CSV", csv, file_name="parkinsons_predictions.csv", mime='text/csv')
        else:
            st.error("❌ CSV must contain all 7 required feature columns.")
    except Exception as e:
        st.error(f"⚠️ Error processing CSV: {e}")
