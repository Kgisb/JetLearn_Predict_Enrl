import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

st.set_page_config(page_title="JetLearn XGBoost Predictor", layout="wide")
st.title("📈 JetLearn Predictor (Trains on Work_JL_DB_Cleaned.csv)")

# === STEP 1: Load and prepare training data ===
@st.cache_data
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    df = df[['Create Date', 'JetLearn Deal Source', 'Country', 'Age', 'HubSpot Deal Score', 'Payment Received Date']].copy()
    df.dropna(subset=['Create Date', 'JetLearn Deal Source', 'Country', 'Age', 'HubSpot Deal Score'], inplace=True)

    df['Create Date'] = pd.to_datetime(df['Create Date'], dayfirst=True, errors='coerce')
    df['Payment Received Date'] = pd.to_datetime(df['Payment Received Date'], dayfirst=True, errors='coerce')

    df['Create_Month'] = df['Create Date'].dt.to_period('M')
    df['Payment_Month'] = df['Payment Received Date'].dt.to_period('M')
    df['Enroll_Same_Month'] = (df['Create_Month'] == df['Payment_Month']).astype(int)

    return df

# === STEP 2: Train XGBoost model ===
@st.cache_resource
def train_model(df):
    df['JetLearn Deal Source'] = df['JetLearn Deal Source'].fillna("Unknown").astype(str)
    df['Country'] = df['Country'].fillna("Unknown").astype(str)

    encoder_deal_source = LabelEncoder()
    encoder_country = LabelEncoder()

    encoder_deal_source.fit(pd.concat([df['JetLearn Deal Source'], pd.Series(["Unknown"])], ignore_index=True))
    encoder_country.fit(pd.concat([df['Country'], pd.Series(["Unknown"])], ignore_index=True))

    df['JetLearn Deal Source'] = encoder_deal_source.transform(df['JetLearn Deal Source'])
    df['Country'] = encoder_country.transform(df['Country'])

    X = df[['JetLearn Deal Source', 'Country', 'Age', 'HubSpot Deal Score']]
    y = df['Enroll_Same_Month']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    return model, encoder_deal_source, encoder_country, df['Age'].median()

# === STEP 3: Encode prediction data ===
def safe_encode(column, encoder, name):
    if column.isnull().any():
        st.warning(f"⚠️ Missing values in '{name}' — replaced with 'Unknown'")
        column = column.fillna("Unknown")
    column = column.astype(str)
    known_classes = list(encoder.classes_)
    column = column.apply(lambda x: x if x in known_classes else "Unknown")
    return encoder.transform(column)

# === STEP 4: Load training data and model ===
training_file = "Work_JL_DB_Cleaned.csv"
try:
    df_model = load_and_prepare_data(training_file)
    model, encoder_deal_source, encoder_country, median_age = train_model(df_model)
    st.success("✅ Model trained using Work_JL_DB_Cleaned.csv")
except Exception as e:
    st.error(f"❌ Failed to train model from {training_file}: {e}")
    st.stop()

# === STEP 5: Upload CSV for prediction ===
st.header("📤 Upload CSV for Prediction")
input_file = st.file_uploader("Upload CSV with: Create Date, JetLearn Deal Source, Country, Age, HubSpot Deal Score", type="csv")

if input_file:
    try:
        df_input = pd.read_csv(input_file)
        df_orig = df_input.copy()

        df_input['JetLearn Deal Source'] = safe_encode(df_input['JetLearn Deal Source'], encoder_deal_source, "JetLearn Deal Source")
        df_input['Country'] = safe_encode(df_input['Country'], encoder_country, "Country")

        df_input['HubSpot Deal Score'] = pd.to_numeric(df_input['HubSpot Deal Score'], errors='coerce')
        if df_input['HubSpot Deal Score'].isnull().any():
            st.warning("⚠️ Missing HubSpot Deal Score — filled with 0")
            df_input['HubSpot Deal Score'] = df_input['HubSpot Deal Score'].fillna(0)
        if (df_input['HubSpot Deal Score'] < 0).any():
            st.info("ℹ️ Negative HubSpot Deal Score treated as valid (e.g., invalid deals)")

        df_input['Age'] = pd.to_numeric(df_input['Age'], errors='coerce')
        if df_input['Age'].isnull().any():
            st.warning("⚠️ Missing Age values — filled with median")
            df_input['Age'] = df_input['Age'].fillna(median_age)

        features = ['JetLearn Deal Source', 'Country', 'Age', 'HubSpot Deal Score']
        predictions = model.predict(df_input[features])
        df_orig['Predicted Enroll (Same Month)'] = predictions

        st.write("📊 Prediction Output", df_orig.head(20))
        csv_out = df_orig.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv_out, "predicted_output.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
