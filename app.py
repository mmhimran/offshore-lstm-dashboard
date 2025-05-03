import streamlit as st
import pandas as pd
import numpy as np
import io
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set page config at the very top!
st.set_page_config(page_title="Offshore LSTM Forecast Dashboard", layout="wide")

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deep_lstm_checkpoint.keras")

model = load_model()

# Excel writer
def generate_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    return output.getvalue()

# Forecasting function
def forecast_temperature(data, lookback=504, steps=168):
    data_values = data[['Te03m', 'Te30m', 'Te50m']].values
    current_input = data_values[-lookback:].reshape(1, lookback, 3)
    predictions = []

    for _ in range(steps):
        pred = model.predict(current_input, verbose=0)[0]
        predictions.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[pred]], axis=1)

    future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(hours=1), periods=steps, freq='H')
    forecast_df = pd.DataFrame(predictions, columns=['Te03m', 'Te30m', 'Te50m'])
    forecast_df.insert(0, 'Date', future_dates)
    return forecast_df

# Streamlit UI
st.set_page_config(page_title="Offshore LSTM Forecast Dashboard", layout="wide")
st.title("📉 Select Forecasting Mode")

mode = st.radio("Choose your desired mode:", [
    "Prediction and Comparison with Actual",
    "Forecasting Only",
    "Compare xlsx file"
])

# === Mode 1: Prediction and Comparison with Actual ===
if mode == "Prediction and Comparison with Actual":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        st.subheader("📋 Uploaded Data")
        st.dataframe(df.tail())

        if st.button("🔮 Run Prediction"):
            forecast_df = forecast_temperature(df)

            result = pd.merge(forecast_df, df[['Date', 'Te03m', 'Te30m', 'Te50m']], on='Date', how='left')
            for col in ['Te03m', 'Te30m', 'Te50m']:
                result[f'Error_{col}'] = result[col] - result[f'{col}_x']
                result[f'Accuracy_{col}'] = 100 - abs(result[f'Error_{col}']) / result[col] * 100
                result.rename(columns={f'{col}_x': f'Actual_{col}', f'{col}_y': f'Predicted_{col}'}, inplace=True)

            # Reorder
            columns_order = ['Date']
            for col in ['Te03m', 'Te30m', 'Te50m']:
                columns_order += [f'Actual_{col}', f'Predicted_{col}', f'Error_{col}', f'Accuracy_{col}']
            result = result[columns_order]

            st.success("✅ Prediction completed.")
            st.download_button("📥 Download Prediction vs Actual Results", generate_excel(result), file_name="forecast_comparison.xlsx")

# === Mode 2: Forecasting Only ===
elif mode == "Forecasting Only":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        st.subheader("📋 Uploaded Data")
        st.dataframe(df.tail())

        if st.button("🔮 Run Forecast Only"):
            forecast = forecast_temperature(df)
            st.success("✅ Forecasting complete.")
            st.download_button("📥 Download Forecast File", generate_excel(forecast), file_name="forecast_only.xlsx")

# === Mode 3: Compare xlsx file ===
elif mode == "Compare xlsx file":
    actual_file = st.file_uploader("Upload Actual File", type=["csv", "xlsx"], key="actual")
    predicted_file = st.file_uploader("Upload Predicted File", type=["csv", "xlsx"], key="predicted")

    if actual_file and predicted_file:
        actual_df = pd.read_excel(actual_file) if actual_file.name.endswith('xlsx') else pd.read_csv(actual_file)
        predicted_df = pd.read_excel(predicted_file) if predicted_file.name.endswith('xlsx') else pd.read_csv(predicted_file)

        actual_df['Date'] = pd.to_datetime(actual_df['Date'])
        predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])

        merged = pd.merge(actual_df, predicted_df, on='Date', how='inner')
        result = pd.DataFrame()
        result['Date'] = merged['Date']

        for col in ['Te03m', 'Te30m', 'Te50m']:
            result[f'Actual_{col}'] = merged[col]
            result[f'Predicted_{col}'] = merged[f'Pred_{col}']
            result[f'Error_{col}'] = result[f'Actual_{col}'] - result[f'Predicted_{col}']
            result[f'Accuracy_{col}'] = 100 - abs(result[f'Error_{col}']) / result[f'Actual_{col}'] * 100

        st.success("✅ Comparison done.")
        st.download_button("📥 Download Comparison Results", generate_excel(result), file_name="comparison_result.xlsx")
