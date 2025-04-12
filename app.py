import streamlit as st
import pandas as pd
import numpy as np
import pickle  # Or joblib, depending on how you saved your model
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import io  # For handling file-like objects
import joblib #For loading the model

# --- Helper functions for downloading files ---
def download_pkl_file(object_to_download, download_filename, download_button_text):
    """
    Generates a download button for a Python object serialized with pickle.

    Args:
        object_to_download:  The Python object to download.
        download_filename (str): The filename for the downloaded file.
        download_button_text (str): The text to display on the download button.
    """
    try:
        # Create a BytesIO object to hold the pickled data
        with io.BytesIO() as buf:
            pickle.dump(object_to_download, buf)
            buf.seek(0)  # Go to the beginning of the buffer
            st.download_button(
                label=download_button_text,
                data=buf,
                file_name=download_filename,
                mime="application/octet-stream",  # Generic binary file type
            )
    except Exception as e:
        st.error(f"Error creating download: {e}")


# --- Main Streamlit app ---
def main():
    st.title("Parkinson's Disease Prediction")

    st.markdown("""
        This app predicts the likelihood of Parkinson's Disease based on vocal measurements.
        **Disclaimer:** This is a screening tool and not a substitute for professional medical advice.
        """)

    # Dark/light mode toggle
    if 'dark_mode' not in st.session_state:
        st.session_state['dark_mode'] = False

    def toggle_dark_mode():
        st.session_state['dark_mode'] = not st.session_state['dark_mode']

    st.sidebar.button("Toggle Dark Mode", on_click=toggle_dark_mode)

    # Apply theme
    if st.session_state['dark_mode']:
        st.markdown(
            """
            <style>
            body {
                color: white;
                background-color: #111;
            }
            .stTextInput > label {
                color: white;
            }
            .stNumberInput > label {
                color: white;
            }
            .stButton > button {
                color: white;
                background-color: #333;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            body {
                color: #111;
                background-color: white;
            }
             .stTextInput > label {
                color: #111;
            }
            .stNumberInput > label {
                color: #111;
            }
           .stButton > button {
                color: #111;
                background-color: #eee;
            }

            </style>
            """,
            unsafe_allow_html=True
        )

    # --- Model Loading and Download ---
    st.header("Model and Preprocessing Files")

    # Load the trained model, scaler, and feature selector
    try:
        model = joblib.load("best_model_XGBoost_ANOVA.pkl") # Ensure this path is correct
        #with open("scaler.pkl", "rb") as f:  # Replace with your scaler file if you saved it separately
        #    scaler = pickle.load(f)
        #with open("feature_selector.pkl", "rb") as f:  # Replace with your feature selector if saved separately
        #    selector = pickle.load(f)
    except FileNotFoundError:
        st.error("Required model or preprocessing files not found. Please ensure they are in the correct location.")
        st.stop()

    download_pkl_file(
        model, "best_model_XGBoost_ANOVA.pkl", "Download Trained Model (.pkl)"
    )
    #download_pkl_file(
    #    scaler, "scaler.pkl", "Download Scaler (.pkl)"
    #)
    #download_pkl_file(
    #    selector, "feature_selector.pkl", "Download Feature Selector (.pkl)"
    #)


    # --- Remainder of your app (Input, Prediction, Feature Info) ---
    st.header("Enter Vocal Measurements")

    # Feature names and descriptions (same as before)
    feature_names = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE",
        "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    feature_descriptions = {
        "MDVP:Fo(Hz)": "Average vocal fundamental frequency",
        "MDVP:Fhi(Hz)": "Maximum vocal fundamental frequency",
        "MDVP:Flo(Hz)": "Minimum vocal fundamental frequency",
        "MDVP:Jitter(%)": "Jitter: variation in fundamental frequency",
        "MDVP:Jitter(Abs)": "Jitter: absolute variation in fundamental frequency",
        "MDVP:RAP": "Relative Amplitude Perturbation",
        "MDVP:PPQ": "Five-point Period Perturbation Quotient",
        "Jitter:DDP": "Jitter:DDP",
        "MDVP:Shimmer": "Shimmer: variation in amplitude",
        "MDVP:Shimmer(dB)": "Shimmer: variation in amplitude in dB",
        "Shimmer:APQ3": "Three-point Amplitude Perturbation Quotient",
        "Shimmer:APQ5": "Five-point Amplitude Perturbation Quotient",
        "MDVP:APQ": "Amplitude Perturbation Quotient",
        "Shimmer:DDA": "Shimmer:DDA",
        "NHR": "Noise-to-Harmonics Ratio",
        "HNR": "Harmonics-to-Noise Ratio",
        "RPDE": "Recurrence Period Density Entropy",
        "DFA": "Signal fractal scaling exponent",
        "spread1": "Nonlinear measure of fundamental frequency variation",
        "spread2": "Nonlinear measure of amplitude variation",
        "D2": "Correlation dimension",
        "PPE": "Pitch Period Entropy"
    }

    # Get the top 15 features (as used in the paper with ANOVA)
    top_15_features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "NHR", "HNR"]  # Replace with actual top 15 if different


    input_data = {}
    st.subheader("Top 15 Features (Required)")
    for feature in top_15_features:
        full_name = feature_descriptions.get(feature, feature)
        input_data[feature] = st.number_input(full_name, key=feature + "_required")

    st.subheader("Optional Features")
    remaining_features = [f for f in feature_names if f not in top_15_features]
    for feature in remaining_features:
        full_name = feature_descriptions.get(feature, feature)
        input_data[feature] = st.number_input(full_name, value=0.0, key=feature + "_optional")

    # Prediction
    if st.button("Predict"):
        # Create DataFrame from input data
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]  # Ensure correct order

        # Preprocess the input data
        #  Replace this with your actual preprocessing and prediction logic
        #X = df.drop(columns='status')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(input_df)
        selector = SelectKBest(score_func=f_classif, k=15)
        X_anova = selector.fit_transform(X_scaled, np.zeros(len(input_df)))  # Dummy y for transform
        input_selected = X_anova
        prediction = model.predict(input_selected)[0]
        probability = model.predict_proba(input_selected)[0][1]


        st.subheader("Prediction Result")
        if prediction == 1:
            st.warning(f"Likely to have Parkinson's Disease (Probability: {probability:.2f})")
        else:
            st.success(f"Unlikely to have Parkinson's Disease (Probability: {probability:.2f})")

    # Feature Information
    st.header("Feature Information")
    for feature, description in feature_descriptions.items():
        st.markdown(f"**{feature}**: {description}")


if __name__ == "__main__":
    main()
