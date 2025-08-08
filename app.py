import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

st.set_page_config(page_title="JetLearn Enrollment Predictor", layout="wide")

st.title("📈 JetLearn Enrollment Predictor")
st.markdown("This tool predicts whether a deal will result in **enrollment in the same month** based on the uploaded data.")

# === STEP 1: Load and clean data ===
@st.cache_data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df = df[['Create Date', 'JetLearn Deal Source', 'Country', 'Age', 'HubSpot Deal Score', 'Payment Received Date']].copy()
    df.dropna(subset=['Create Date', 'JetLearn Deal Source', 'Country', 'Age', 'HubSpot Deal Score'], inplace=True)

    df['Create Date'] = pd.to_datetime(df['Create Date'], dayfirst=True, errors='coerce')
    df['Payment Received Date'] = pd.to_datetime(df['Payment Received Date'], dayfirst=True, errors='coerce')

    df['Create_Month'] = df['Create Date'].dt.to_period('M')
    df['Payment_Month'] = df['Payment Received Date'].dt.to_period('M')
    df['Enroll_Same_Month'] = (df['Create_Month'] == df['Payment_Month']).astype(int)

    return df

# === STEP 2: Train the model ===
@st.cache_resource
def train_model(df):
    encoder_deal_source = LabelEncoder()
    encoder_country = LabelEncoder()

    df['JetLearn Deal Source'] = encoder_deal_source.fit_transform(df['JetLearn Deal Source'])
    df['Country'] = encoder_country.fit_transform(df['Country'])

    X = df[['JetLearn Deal Source', 'Country', 'Age', 'HubSpot Deal Score']]
    y = df['Enroll_Same_Month']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    return model, encoder_deal_source, encoder_country

# === STEP 3: Load + Train ===
data_file_path = "Work_JL_DB_Cleaned.csv"
if not os.path.exists(data_file_path):
    st.error("❌ 'Work_JL_DB_Cleaned.csv' not found. Please make sure it's in the same folder as this app.")
    st.stop()

df_model = load_and_prepare_data(data_file_path)
model, encoder_deal_source, encoder_country = train_model(df_model)

st.success("✅ Model trained successfully using Work_JL_DB_Cleaned.csv")

# === STEP 4: Upload new data for prediction ===
st.header("📤 Upload File for Prediction")
input_file = st.file_uploader("Upload a CSV with: Create Date, JetLearn Deal Source, Country, Age, HubSpot Deal Score", type="csv")

if input_file:
    try:
        df_input = pd.read_csv(input_file)
        df_orig = df_input.copy()

        df_input['JetLearn Deal Source'] = encoder_deal_source.transform(df_input['JetLearn Deal Source'])
        df_input['Country'] = encoder_country.transform(df_input['Country'])

        features = ['JetLearn Deal Source', 'Country', 'Age', 'HubSpot Deal Score']
        predictions = model.predict(df_input[features])

        df_orig['Predicted Enroll (Same Month)'] = predictions

        st.write("📊 Prediction Output", df_orig.head(20))
        csv_out = df_orig.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv_out, "predicted_output.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")