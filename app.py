import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Forecasting Dashboard", layout="centered")
st.markdown("### 🧰 Select Forecasting Mode")

mode = st.radio("Choose your desired mode:", (
    "Prediction and Comparison with Actual",
    "Forecasting Only",
    "Compare xlsx file"
))

# === Excel file generator ===
def generate_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    output.seek(0)
    return output

# === Mode 1: Prediction and Comparison with Actual ===
if mode == "Prediction and Comparison with Actual":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.markdown("### 🧾 Uploaded Data")
        st.dataframe(df)

        # Dummy prediction (replace with real model)
        forecast = df.copy()
        for col in ['Te03m', 'Te30m', 'Te50m']:
            forecast[f'Predicted_{col}'] = forecast[col] + np.random.normal(0, 0.1, len(forecast))
            forecast[f'Error_{col}'] = forecast[f'Predicted_{col}'] - forecast[col]
            forecast[f'Accuracy_{col} (%)'] = 100 - abs(forecast[f'Error_{col}']) / forecast[col] * 100

        # Rearranged columns
        result = pd.DataFrame()
        result['Date'] = forecast['Date']
        for col in ['Te03m', 'Te30m', 'Te50m']:
            result[f'Actual_{col}'] = forecast[col]
            result[f'Predicted_{col}'] = forecast[f'Predicted_{col}']
            result[f'Error_{col}'] = forecast[f'Error_{col}']
            result[f'Accuracy_{col} (%)'] = forecast[f'Accuracy_{col} (%)']

        excel_data = generate_excel(result)
        st.download_button("📥 Download Prediction vs Actual Results", excel_data, file_name="prediction_vs_actual.xlsx")

# === Mode 2: Forecasting Only ===
elif mode == "Forecasting Only":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=['csv', 'xlsx'])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.markdown("### 🧾 Uploaded Data")
        st.dataframe(df)

        # Dummy forecast (replace with real LSTM forecast)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            df[f'Forecasted_{col}'] = df[col] + np.random.normal(0, 0.2, len(df))

        result = df[['Date'] + [f'Forecasted_{col}' for col in ['Te03m', 'Te30m', 'Te50m']]]
        excel_data = generate_excel(result)
        st.download_button("📥 Download Forecasting Results", excel_data, file_name="forecasting_only.xlsx")

# === Mode 3: Compare xlsx file (actual vs predicted) ===
elif mode == "Compare xlsx file":
    actual_file = st.file_uploader("Upload Actual File", type=['csv', 'xlsx'], key='actual')
    predicted_file = st.file_uploader("Upload Predicted File", type=['csv', 'xlsx'], key='predicted')

    if actual_file and predicted_file:
        actual_df = pd.read_excel(actual_file)
        pred_df = pd.read_excel(predicted_file)

        merged = pd.merge(actual_df, pred_df, on='Date', how='inner')
        result = pd.DataFrame()
        result['Date'] = merged['Date']

        for col in ['Te03m', 'Te30m', 'Te50m']:
            result[f'Actual_{col}'] = merged[col]
            result[f'Predicted_{col}'] = merged[f'Pred_{col}']
            result[f'Error_{col}'] = result[f'Predicted_{col}'] - result[f'Actual_{col}']
            result[f'Accuracy_{col} (%)'] = 100 - abs(result[f'Error_{col}']) / result[f'Actual_{col}'] * 100

        excel_data = generate_excel(result)
        st.download_button("📥 Download Comparison Results", excel_data, file_name="comparison_result.xlsx")
