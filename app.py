import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="JetLearn Monthly Enrollment Predictor", layout="wide")
st.title("📊 JetLearn Monthly Enrollment Predictor (LightGBM)")

# === Load and prepare training data ===
@st.cache_data
def load_data():
    df = pd.read_csv("Work_JL_DB_Cleaned.csv")
    df.columns = df.columns.str.strip()

    df = df[['Create Date', 'JetLearn Deal Source', 'Country', 'Age', 'HubSpot Deal Score', 'Payment Received Date']].copy()
    df.dropna(subset=['Create Date', 'JetLearn Deal Source', 'Country', 'Age', 'HubSpot Deal Score'], inplace=True)

    df['Create Date'] = pd.to_datetime(df['Create Date'], dayfirst=True, errors='coerce')
    df['Payment Received Date'] = pd.to_datetime(df['Payment Received Date'], dayfirst=True, errors='coerce')

    df['Create_Month'] = df['Create Date'].dt.to_period('M')
    df['Payment_Month'] = df['Payment Received Date'].dt.to_period('M')
    df['Enroll_Same_Month'] = (df['Create_Month'] == df['Payment_Month']).astype(int)

    return df

df_model = load_data()

# === Split data: Train before Aug 2025, Test on Aug 2025 ===
holdout_month = pd.Period("2025-08")
df_train = df_model[df_model['Create_Month'] < holdout_month].copy()
df_test = df_model[df_model['Create_Month'] == holdout_month].copy()

# === Encode categorical ===
def encode_features(train, test):
    train['JetLearn Deal Source'] = train['JetLearn Deal Source'].fillna("Unknown").astype(str)
    train['Country'] = train['Country'].fillna("Unknown").astype(str)
    test['JetLearn Deal Source'] = test['JetLearn Deal Source'].fillna("Unknown").astype(str)
    test['Country'] = test['Country'].fillna("Unknown").astype(str)

    enc_deal = LabelEncoder()
    enc_country = LabelEncoder()

    enc_deal.fit(pd.concat([train['JetLearn Deal Source'], pd.Series(["Unknown"])], ignore_index=True))
    enc_country.fit(pd.concat([train['Country'], pd.Series(["Unknown"])], ignore_index=True))

    train['JetLearn Deal Source'] = enc_deal.transform(train['JetLearn Deal Source'])
    train['Country'] = enc_country.transform(train['Country'])

    test['JetLearn Deal Source'] = test['JetLearn Deal Source'].apply(lambda x: x if x in enc_deal.classes_ else "Unknown")
    test['Country'] = test['Country'].apply(lambda x: x if x in enc_country.classes_ else "Unknown")

    test['JetLearn Deal Source'] = enc_deal.transform(test['JetLearn Deal Source'])
    test['Country'] = enc_country.transform(test['Country'])

    return train, test

df_train, df_test = encode_features(df_train, df_test)

# === Train Model ===
features = ['JetLearn Deal Source', 'Country', 'Age', 'HubSpot Deal Score']
X_train = df_train[features]
y_train = df_train['Enroll_Same_Month']
X_test = df_test[features]
y_test = df_test['Enroll_Same_Month']

model = LGBMClassifier(random_state=42, n_estimators=300, learning_rate=0.05)
model.fit(X_train, y_train)

# === Predict on Aug 2025 ===
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_prob > 0.5).astype(int)

# === Output metrics ===
actual = int(y_test.sum())
predicted = round(y_pred_prob.sum(), 2)
acc = round(accuracy_score(y_test, y_pred_class) * 100, 2)
auc = round(roc_auc_score(y_test, y_pred_prob) * 100, 2)

st.subheader("📅 August 2025 Enrollment Prediction Results")
st.markdown(f"""
- ✅ **Actual Enrollments**: `{actual}`
- 📊 **Predicted Enrollments (sum of probabilities)**: `{predicted}`
- 🎯 **Classification Accuracy**: `{acc}%`
- 🔍 **ROC AUC Score**: `{auc}%`
""")

st.markdown("---")
st.subheader("📥 Export")
st.markdown("Model does row-wise predictions and aggregates them for monthly forecast.")
