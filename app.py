import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model
@st.cache_resource
def load_model():
    model = joblib.load('best_model_Random Forest_ANOVA.pkl')
    return model

model = load_model()

# Feature descriptions
feature_descriptions = {
    'MDVP:Fo(Hz)': 'Average vocal fundamental frequency',
    'MDVP:Fhi(Hz)': 'Maximum vocal fundamental frequency',
    'MDVP:Flo(Hz)': 'Minimum vocal fundamental frequency',
    'MDVP:Jitter(%)': 'Measure of variation in fundamental frequency',
    'MDVP:Jitter(Abs)': 'Absolute measure of jitter',
    'MDVP:RAP': 'Relative amplitude perturbation',
    'MDVP:PPQ': 'Five-point period perturbation quotient',
    'Jitter:DDP': 'Average absolute difference of differences between jitter cycles',
    'MDVP:Shimmer': 'Measure of variation in amplitude',
    'MDVP:Shimmer(dB)': 'Shimmer in decibels',
    'Shimmer:APQ3': 'Three-point amplitude perturbation quotient',
    'Shimmer:APQ5': 'Five-point amplitude perturbation quotient',
    'MDVP:APQ': 'Amplitude perturbation quotient',
    'Shimmer:DDA': 'Average absolute difference between consecutive differences of amplitude',
    'NHR': 'Noise-to-harmonics ratio',
    'HNR': 'Harmonics-to-noise ratio',
    'RPDE': 'Recurrence period density entropy measure',
    'DFA': 'Signal fractal scaling exponent',
    'spread1': 'Nonlinear measure of fundamental frequency variation',
    'spread2': 'Nonlinear measure of fundamental frequency variation',
    'D2': 'Nonlinear dynamical complexity measure',
    'PPE': 'Pitch period entropy'
}

# Top 10 most important features from the model
top_features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)'
]

# Create the Streamlit app
def main():
    st.title("Parkinson's Disease Prediction")
    st.markdown("""
    This app predicts the likelihood of Parkinson's disease based on voice measurements.
    The model was trained using the top 10 most important features from voice recordings.
    """)
    
    st.sidebar.header("About")
    st.sidebar.info("""
    This prediction model uses a Random Forest classifier trained on the UCI Parkinson's dataset.
    It analyzes voice measurements to predict the presence of Parkinson's disease.
    """)
    
    st.sidebar.header("Feature Descriptions")
    selected_feature = st.sidebar.selectbox("Select a feature to view description:", top_features)
    st.sidebar.write(feature_descriptions[selected_feature])
    
    st.header("Input Voice Measurements")
    
    # Create input fields for the top 10 features
    inputs = {}
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['MDVP:Fo(Hz)'] = st.number_input('Average vocal fundamental frequency (Hz)', 
                                              min_value=80.0, max_value=300.0, value=120.0, step=0.1)
        inputs['MDVP:Fhi(Hz)'] = st.number_input('Maximum vocal fundamental frequency (Hz)', 
                                               min_value=100.0, max_value=600.0, value=150.0, step=0.1)
        inputs['MDVP:Flo(Hz)'] = st.number_input('Minimum vocal fundamental frequency (Hz)', 
                                               min_value=60.0, max_value=250.0, value=75.0, step=0.1)
        inputs['MDVP:Jitter(%)'] = st.number_input('Jitter percentage', 
                                                 min_value=0.0, max_value=0.1, value=0.005, step=0.0001, format="%.4f")
        inputs['MDVP:Jitter(Abs)'] = st.number_input('Absolute jitter', 
                                                   min_value=0.0, max_value=0.001, value=0.00005, step=0.00001, format="%.5f")
    
    with col2:
        inputs['MDVP:RAP'] = st.number_input('Relative amplitude perturbation', 
                                           min_value=0.0, max_value=0.1, value=0.003, step=0.0001, format="%.4f")
        inputs['MDVP:PPQ'] = st.number_input('Five-point period perturbation quotient', 
                                           min_value=0.0, max_value=0.1, value=0.005, step=0.0001, format="%.4f")
        inputs['Jitter:DDP'] = st.number_input('Average absolute difference of jitter cycles', 
                                             min_value=0.0, max_value=0.1, value=0.01, step=0.0001, format="%.4f")
        inputs['MDVP:Shimmer'] = st.number_input('Shimmer', 
                                               min_value=0.0, max_value=0.2, value=0.03, step=0.001, format="%.3f")
        inputs['MDVP:Shimmer(dB)'] = st.number_input('Shimmer in decibels', 
                                                   min_value=0.0, max_value=2.0, value=0.3, step=0.01)
    
    # Create a button for prediction
    if st.button('Predict'):
        # Create a DataFrame from the inputs
        input_df = pd.DataFrame([inputs])
        
        # Scale the features (using the same scaler from training)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(input_df)
        
        # Make prediction
        prediction = model.predict(X_scaled)
        prediction_proba = model.predict_proba(X_scaled)
        
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"The model predicts **Parkinson's disease is likely** (probability: {prediction_proba[0][1]:.2%})")
        else:
            st.success(f"The model predicts **Parkinson's disease is unlikely** (probability: {prediction_proba[0][0]:.2%})")
        
        st.subheader("Detailed Probabilities")
        proba_df = pd.DataFrame({
            'Condition': ['Healthy', 'Parkinson\'s'],
            'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
        })
        st.bar_chart(proba_df.set_index('Condition'))
    
    st.markdown("""
    ---
    **Note:** This tool is for informational purposes only and should not replace professional medical advice.
    """)

if __name__ == '__main__':
    main()
