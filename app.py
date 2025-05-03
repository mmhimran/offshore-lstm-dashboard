import streamlit as st
import pandas as pd
import numpy as np
import io
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

# ⚙️ MUST be first Streamlit command
st.set_page_config(page_title="Offshore LSTM Forecast Dashboard", layout="wide")

# Load LSTM model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deep_lstm_checkpoint.keras")

model = load_model()

# Excel download utility
def generate_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast')
    return output.getvalue()

# Accuracy utility
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
        actual_col = col
        pred_col = f'Pred_{col}'
        if actual_col in merged.columns and pred_col in merged.columns:
            result[f'Actual_{col}'] = merged[actual_col]
            result[f'Predicted_{col}'] = merged[pred_col]
            result[f'Error_{col}'] = merged[actual_col] - merged[pred_col]
            result[f'Accuracy_{col}'] = 100 - (np.abs(result[f'Error_{col}']) / merged[actual_col] * 100)
    return result

# ======================== UI ========================
st.title("🧰 Select Forecasting Mode")
mode = st.radio("Choose your desired mode:", [
    "Prediction and Comparison with Actual",
    "Forecasting Only",
    "Compare xlsx file"
])

# ----------------- Option 1: Prediction & Comparison with Actual -----------------
if mode == "Prediction and Comparison with Actual":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])

        forecast = forecast_temperature(df)
        st.subheader("📋 Uploaded Data")
        st.dataframe(df.tail())

        actual = df[['Date', 'Te03m', 'Te30m', 'Te50m']].iloc[-168:].reset_index(drop=True)
        predicted = forecast.copy()
        predicted.columns = ['Date', 'Pred_Te03m', 'Pred_Te30m', 'Pred_Te50m']

        comparison = compare_data(actual, predicted)

        st.download_button("📥 Download Prediction vs Actual Results", generate_excel(comparison), file_name="comparison_results.xlsx")

# ----------------- Option 2: Forecasting Only -----------------
elif mode == "Forecasting Only":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])

        forecast = forecast_temperature(df)
        st.subheader("📋 Uploaded Data")
        st.dataframe(df.tail())

        st.download_button("📥 Download Forecast", generate_excel(forecast), file_name="forecast_results.xlsx")

# ----------------- Option 3: Compare Two Excel Files -----------------
elif mode == "Compare xlsx file":
    st.markdown("#### Upload Actual File")
    actual_file = st.file_uploader("Upload Actual File", type=['csv', 'xlsx'], key="actual")
    st.markdown("#### Upload Predicted File")
    predicted_file = st.file_uploader("Upload Predicted File", type=['csv', 'xlsx'], key="predicted")

    if actual_file and predicted_file:
        actual_df = pd.read_excel(actual_file) if actual_file.name.endswith('xlsx') else pd.read_csv(actual_file)
        predicted_df = pd.read_excel(predicted_file) if predicted_file.name.endswith('xlsx') else pd.read_csv(predicted_file)

        actual_df['Date'] = pd.to_datetime(actual_df['Date'])
        predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])

        predicted_df.rename(columns=lambda x: f'Pred_{x}' if x != 'Date' else x, inplace=True)

        result = compare_data(actual_df, predicted_df)

        st.download_button("📥 Download Comparison Results", generate_excel(result), file_name="comparison_only.xlsx")
