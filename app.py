# Full Updated app.py with PETRONAS-style UI and Five Features

import streamlit as st
import pandas as pd
import numpy as np
import io
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

# Set page config
st.set_page_config(
    page_title="OFFSHORE TEMPERATURE FORECAST",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for PETRONAS look
st.markdown("""
    <style>
        .main {
            background-color: #F4FCFD;
        }
        .css-18e3th9 {
            background-color: #0FB7A8 !important;
            color: white;
        }
        .css-1d391kg {
            background-color: #F4FCFD !important;
        }
        .block-container {
            padding: 2rem;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #0FB7A8;
        }
    </style>
""", unsafe_allow_html=True)

# Load LSTM model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deep_lstm_checkpoint.keras")

model = load_model()

# Excel export helper
def generate_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast')
    return output.getvalue()

# Accuracy computation
def compute_accuracy(y_true, y_pred):
    return np.mean(100 - (np.abs(y_true - y_pred) / y_true * 100))

# Forecast function
def forecast_temperature(data, lookback=504, forecast_steps=168):
    values = data[['Te03m', 'Te30m', 'Te50m']].values
    scaled = (values - values.mean(axis=0)) / values.std(axis=0)
    input_seq = scaled[-lookback:].reshape(1, lookback, 3)
    preds = []
    for _ in range(forecast_steps):
        pred = model.predict(input_seq, verbose=0)
        preds.append(pred[0])
        input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 3), axis=1)
    preds = np.array(preds)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    unscaled = preds * std + mean
    dates = pd.date_range(start=data['Date'].iloc[-1] + timedelta(hours=1), periods=forecast_steps, freq='H')
    forecast_df = pd.DataFrame(unscaled, columns=['Te03m', 'Te30m', 'Te50m'])
    forecast_df.insert(0, 'Date', dates)
    return forecast_df

# Comparison function
def compare_data(actual_df, predicted_df):
    merged = pd.merge(actual_df, predicted_df, on='Date')
    result = pd.DataFrame({'Date': merged['Date']})
    for col in ['Te03m', 'Te30m', 'Te50m']:
        result[f'Actual_{col}'] = merged[col]
        result[f'Predicted_{col}'] = merged[f'Pred_{col}']
        result[f'Error_{col}'] = result[f'Actual_{col}'] - result[f'Predicted_{col}']
        result[f'Accuracy_{col}'] = 100 - (np.abs(result[f'Error_{col}']) / result[f'Actual_{col}'] * 100)
    return result

# ========================== UI ==============================
st.title("🧰 OFFSHORE TEMPERATURE FORECAST")
mode = st.sidebar.radio("Choose your desired mode:", [
    "Prediction and Comparison with Given Actual Value",
    "Future Forecasting Only",
    "Compare Predicted with Actual",
    "Visualize Actual vs Predicted",
    "Visualize Accuracy"
])

# === 1. Prediction and Comparison ===
if mode == "Prediction and Comparison with Actual":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=['csv', 'xlsx'])
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        lookback_df = df.iloc[:504]
        actual_df = df.iloc[504:]
        forecast = forecast_temperature(lookback_df)
        st.subheader("📋 Uploaded Lookback Data")
        st.dataframe(lookback_df.tail())
        predicted = forecast.rename(columns={
            'Te03m': 'Pred_Te03m',
            'Te30m': 'Pred_Te30m',
            'Te50m': 'Pred_Te50m'
        })
        actual_df = actual_df[['Date', 'Te03m', 'Te30m', 'Te50m']].reset_index(drop=True)
        predicted = predicted[['Date', 'Pred_Te03m', 'Pred_Te30m', 'Pred_Te50m']].reset_index(drop=True)
        merged = pd.merge(actual_df, predicted, on='Date')
        result = compare_data(actual_df, predicted)
        st.download_button("📥 Download Comparison Results", generate_excel(result), file_name="comparison_results.xlsx")

# === 2. Forecasting Only ===
elif mode == "Forecasting Only":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=['csv', 'xlsx'])
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        forecast = forecast_temperature(df)
        st.subheader("📋 Uploaded Data")
        st.dataframe(df.tail())
        st.download_button("📥 Download Forecast", generate_excel(forecast), file_name="forecast_results.xlsx")

# === 3. Compare XLSX ===
elif mode == "Compare xlsx file":
    actual_file = st.file_uploader("Upload Actual File", type=['csv', 'xlsx'], key="actual")
    predicted_file = st.file_uploader("Upload Predicted File", type=['csv', 'xlsx'], key="predicted")
    if actual_file and predicted_file:
        actual_df = pd.read_excel(actual_file) if actual_file.name.endswith('xlsx') else pd.read_csv(actual_file)
        predicted_df = pd.read_excel(predicted_file) if predicted_file.name.endswith('xlsx') else pd.read_csv(predicted_file)
        actual_df['Date'] = pd.to_datetime(actual_df['Date'])
        predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])
        predicted_df.rename(columns=lambda x: f'Pred_{x}' if x != 'Date' else x, inplace=True)
        result = compare_data(actual_df, predicted_df)
        st.download_button("📥 Download Comparison Results", generate_excel(result), file_name="comparison_only.xlsx")

# === 4. Visualize Actual vs Predicted ===
elif mode == "Visualize Actual vs Predicted":
    file = st.file_uploader("Upload Excel result file", type=['xlsx'], key="vis1")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Te03m', 'Te30m', 'Te50m']:
            plt.figure(figsize=(14, 6))
            plt.plot(df['Date'], df[f'Actual_{col}'], label='Actual', linewidth=3)
            plt.plot(df['Date'], df[f'Predicted_{col}'], label='Predicted', linewidth=3)
            plt.xlabel("Date", fontsize=24, fontname="Times New Roman")
            plt.ylabel("Temperature (°C)", fontsize=24, fontname="Times New Roman")
            plt.title(f"Actual vs Predicted for {col} Temperature", fontsize=30, fontweight='bold', fontname="Times New Roman")
            plt.xticks(fontsize=20, fontname="Times New Roman", rotation=30)
            plt.yticks(fontsize=20, fontname="Times New Roman")
            plt.legend(fontsize=20)
            plt.grid(True)
            st.pyplot(plt)

# === 5. Visualize Accuracy ===
elif mode == "Visualize Accuracy":
    file = st.file_uploader("Upload Excel result file", type=['xlsx'], key="vis2")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Te03m', 'Te30m', 'Te50m']:
            plt.figure(figsize=(14, 6))
            plt.plot(df['Date'], df[f'Accuracy_{col}'], label=f'Accuracy {col}', linewidth=3)
            plt.xlabel("Date", fontsize=24, fontname="Times New Roman")
            plt.ylabel("Accuracy (%)", fontsize=24, fontname="Times New Roman")
            plt.title(f"Accuracy for {col} Temperature", fontsize=30, fontweight='bold', fontname="Times New Roman")
            plt.xticks(fontsize=20, fontname="Times New Roman", rotation=30)
            plt.yticks(fontsize=20, fontname="Times New Roman")
            plt.legend(fontsize=20)
            plt.grid(True)
            st.pyplot(plt)
