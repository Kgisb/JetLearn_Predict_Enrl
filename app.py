import streamlit as st
import pandas as pd
import joblib

from xgboost import XGBClassifier

st.set_page_config(page_title="JetLearn XGBoost Predictor", layout="wide")
st.title("📈 JetLearn Enrollment Predictor (XGBoost)")

# === Load model and encoders ===
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load("xgb_enrollment_model.joblib")
    encoder_deal_source = joblib.load("encoder_deal_source.joblib")
    encoder_country = joblib.load("encoder_country.joblib")
    return model, encoder_deal_source, encoder_country

model, encoder_deal_source, encoder_country = load_model_and_encoders()

# === Safe encoding for categorical columns ===
def safe_encode(column, encoder, name):
    if column.isnull().any():
        st.warning(f"⚠️ Missing values in '{name}' — replaced with 'Unknown'")
        column = column.fillna("Unknown")
    column = column.astype(str)
    known_classes = list(encoder.classes_)
    column = column.apply(lambda x: x if x in known_classes else "Unknown")
    return encoder.transform(column)

# === Upload & Predict ===
st.header("📤 Upload CSV to Predict Enrollments")
uploaded_file = st.file_uploader("Upload CSV with: Create Date, JetLearn Deal Source, Country, Age, HubSpot Deal Score", type="csv")

if uploaded_file:
    try:
        df_input = pd.read_csv(uploaded_file)
        df_orig = df_input.copy()

        # --- Encode categorical ---
        df_input['JetLearn Deal Source'] = safe_encode(df_input['JetLearn Deal Source'], encoder_deal_source, "JetLearn Deal Source")
        df_input['Country'] = safe_encode(df_input['Country'], encoder_country, "Country")

        # --- Handle HubSpot Deal Score ---
        df_input['HubSpot Deal Score'] = pd.to_numeric(df_input['HubSpot Deal Score'], errors='coerce')
        if df_input['HubSpot Deal Score'].isnull().any():
            st.warning("⚠️ Missing HubSpot Deal Score values — filled with 0")
            df_input['HubSpot Deal Score'] = df_input['HubSpot Deal Score'].fillna(0)
        if (df_input['HubSpot Deal Score'] < 0).any():
            st.info("ℹ️ Negative HubSpot Deal Score values detected — treated as valid (e.g., invalid deals)")

        # --- Handle Age ---
        df_input['Age'] = pd.to_numeric(df_input['Age'], errors='coerce')
        if df_input['Age'].isnull().any():
            median_age = df_input['Age'].median()
            st.warning("⚠️ Missing Age values — filled with median")
            df_input['Age'] = df_input['Age'].fillna(median_age)

        # --- Predict ---
        features = ['JetLearn Deal Source', 'Country', 'Age', 'HubSpot Deal Score']
        predictions = model.predict(df_input[features])
        df_orig['Predicted Enroll (Same Month)'] = predictions

        # --- Show & Download ---
        st.write("📊 Prediction Output", df_orig.head(20))
        csv_out = df_orig.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv_out, "predicted_output.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
