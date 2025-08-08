import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from datetime import datetime

st.set_page_config(page_title="JetLearn Enrollments Predictor", layout="wide")
st.title("📊 JetLearn Enrollments Predictor (LightGBM - Monthly Forecast)")

# Load saved model and encoders
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model/xgb_enrollment_model.joblib")
    enc_deal = joblib.load("xgb_model/encoder_deal_source.joblib")
    enc_country = joblib.load("xgb_model/encoder_country.joblib")
    return model, enc_deal, enc_country

model, enc_deal, enc_country = load_model()

uploaded_file = st.file_uploader("📤 Upload CSV with: Create Date, JetLearn Deal Source, Country, Age, HubSpot Deal Score", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    required_cols = ['Create Date', 'JetLearn Deal Source', 'Country', 'Age', 'HubSpot Deal Score']
    if not all(col in df.columns for col in required_cols):
        st.error("❌ CSV must contain all required columns.")
    else:
        # Preprocess input
        df['Create Date'] = pd.to_datetime(df['Create Date'], errors='coerce', dayfirst=True)
        df['Month'] = df['Create Date'].dt.to_period("M")

        df['JetLearn Deal Source'] = df['JetLearn Deal Source'].fillna("Unknown").astype(str)
        df['Country'] = df['Country'].fillna("Unknown").astype(str)
        df['HubSpot Deal Score'] = df['HubSpot Deal Score'].fillna(df['HubSpot Deal Score'].median())
        df['Age'] = df['Age'].fillna(df['Age'].median())

        df['JetLearn Deal Source'] = df['JetLearn Deal Source'].apply(lambda x: x if x in enc_deal.classes_ else "Unknown")
        df['Country'] = df['Country'].apply(lambda x: x if x in enc_country.classes_ else "Unknown")

        df['JetLearn Deal Source'] = enc_deal.transform(df['JetLearn Deal Source'])
        df['Country'] = enc_country.transform(df['Country'])

        # Predict
        features = ['JetLearn Deal Source', 'Country', 'Age', 'HubSpot Deal Score']
        df['Enroll_Prob'] = model.predict_proba(df[features])[:, 1]

        # Monthly prediction aggregation
        monthly_pred = df.groupby('Month')['Enroll_Prob'].sum().reset_index()
        monthly_pred.columns = ['Month', 'Predicted Enrollments']

        st.subheader("📅 Monthly Predicted Enrollments")
        st.dataframe(monthly_pred)

        st.markdown("---")
        st.subheader("📥 Row-wise Prediction")
        st.dataframe(df[['Create Date', 'Age', 'HubSpot Deal Score', 'Enroll_Prob']])
