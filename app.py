import streamlit as st
import pandas as pd
import numpy as np
import io
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
import matplotlib.pyplot as plt

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
        result[f'Actual_{col}'] = merged[col]
        result[f'Predicted_{col}'] = merged[f'Pred_{col}']
        result[f'Error_{col}'] = result[f'Actual_{col}'] - result[f'Predicted_{col}']
        result[f'Accuracy_{col}'] = 100 - (np.abs(result[f'Error_{col}']) / result[f'Actual_{col}'] * 100)
    return result

# ======================== UI ========================
st.title("🧰 Select Forecasting Mode")
mode = st.radio("Choose your desired mode:", [
    "Prediction and Comparison with Actual",
    "Forecasting Only",
    "Compare xlsx file",
    "Plot Actual vs Predicted",
    "Plot Accuracy"
])

# ✅ Option 1
if mode == "Prediction and Comparison with Actual":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=['csv', 'xlsx'])

    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])

        lookback_df = df.iloc[:504]
        actual_df = df.iloc[504:]

        forecast = forecast_temperature(lookback_df)

        st.subheader("📋 Uploaded Data (last few rows from lookback)")
        st.dataframe(lookback_df.tail())

        predicted = forecast.rename(columns={
            'Te03m': 'Pred_Te03m',
            'Te30m': 'Pred_Te30m',
            'Te50m': 'Pred_Te50m'
        })

        actual_df = actual_df[['Date', 'Te03m', 'Te30m', 'Te50m']].reset_index(drop=True)
        predicted = predicted[['Date', 'Pred_Te03m', 'Pred_Te30m', 'Pred_Te50m']].reset_index(drop=True)
        merged = pd.merge(actual_df, predicted, on='Date')

        result = pd.DataFrame({'Date': merged['Date']})
        for col in ['Te03m', 'Te30m', 'Te50m']:
            result[f'Actual_{col}'] = merged[col]
            result[f'Predicted_{col}'] = merged[f'Pred_{col}']
            result[f'Error_{col}'] = result[f'Actual_{col}'] - result[f'Predicted_{col}']
            result[f'Accuracy_{col}'] = 100 - (np.abs(result[f'Error_{col}']) / result[f'Actual_{col}'] * 100)

        result = result[[
            'Date',
            'Actual_Te03m', 'Predicted_Te03m', 'Error_Te03m', 'Accuracy_Te03m',
            'Actual_Te30m', 'Predicted_Te30m', 'Error_Te30m', 'Accuracy_Te30m',
            'Actual_Te50m', 'Predicted_Te50m', 'Error_Te50m', 'Accuracy_Te50m'
        ]]

        st.download_button("📥 Download Prediction vs Actual Results", generate_excel(result), file_name="comparison_results.xlsx")

# ✅ Option 2
elif mode == "Forecasting Only":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])

        forecast = forecast_temperature(df)
        st.subheader("📋 Uploaded Data")
        st.dataframe(df.tail())

        st.download_button("📥 Download Forecast", generate_excel(forecast), file_name="forecast_results.xlsx")

# ✅ Option 3
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

# ✅ Option 4 – Plot Actual vs Predicted
elif mode == "Plot Actual vs Predicted":
    uploaded_file = st.file_uploader("Upload XLSX file with prediction results", type=['xlsx'], key="plot_actual_vs_predicted")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 30})
        for depth in ['Te03m', 'Te30m', 'Te50m']:
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(df['Date'], df[f'Actual_{depth}'], label='Actual', linewidth=4)
            ax.plot(df['Date'], df[f'Predicted_{depth}'], label='Predicted', linewidth=4)
            ax.set_title(f'{depth} Temperature: Actual vs Predicted', fontweight='bold')
            ax.legend(fontsize=28)
            ax.grid(True)
            st.pyplot(fig)

# ✅ Option 5 – Plot Accuracy
elif mode == "Plot Accuracy":
    uploaded_file = st.file_uploader("Upload XLSX file with accuracy results", type=['xlsx'], key="plot_accuracy")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])

        plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 30})
        for depth in ['Te03m', 'Te30m', 'Te50m']:
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(df['Date'], df[f'Accuracy_{depth}'], label=f'Accuracy {depth}', linewidth=4)
            ax.set_title(f'Accuracy for {depth} Temperature', fontweight='bold')
            ax.legend(fontsize=28)
            ax.grid(True)
            st.pyplot(fig)
