# ============================================================
# FULL STREAMLIT DASHBOARD FOR OFFSHORE TEMPERATURE FORECAST
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import tensorflow as tf
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import timedelta

# --- Page Setup ---
st.set_page_config(page_title="OFFSHORE TEMPERATURE FORECAST", layout="wide")

# --- Custom UI Styling ---
st.markdown("""
<style>
    .main, .block-container { background-color: #417C7B !important; }
    h1 { color: #FFFAFA !important; font-size: 40px; font-weight: bold; }
    .stRadio > div { font-size: 18px !important; color: white; }
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🧰 OFFSHORE TEMPERATURE FORECAST")

# --- Mode Selection ---
mode = st.sidebar.radio("Choose your desired mode:", [
    "Prediction and Comparison with Given Actual Value",
    "Future Forecasting Only",
    "Compare Predicted with Actual",
    "Visualize Actual vs Predicted",
    "Visualize Accuracy",
    "Compute SE and MAPE for Each Row",
    "Calculate Overall Metrics",
    "Visualize Error Metrics"
])

# --- Model Loading ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deep_lstm_checkpoint.keras")

model = load_model()

# --- Helper Functions ---
def clean_and_round(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df.round(2)

def forecast_temperature(data, lookback=504, forecast_steps=168):
    values = data[['Te03m', 'Te30m', 'Te50m']].values
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    scaled = (values - mean) / std
    input_seq = scaled[-lookback:].reshape(1, lookback, 3)
    preds = []

    for _ in range(forecast_steps):
        pred = model.predict(input_seq, verbose=0)
        preds.append(pred[0])
        input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 3), axis=1)

    unscaled = np.array(preds) * std + mean
    forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + timedelta(hours=1), periods=forecast_steps, freq='H')
    forecast_df = pd.DataFrame(unscaled, columns=['Te03m', 'Te30m', 'Te50m'])
    forecast_df.insert(0, 'Date', forecast_dates)
    return forecast_df.round(2)

def plot_interactive(df, x_col, y_cols, title, y_title, filename):
    fig = px.line(df, x=x_col, y=y_cols, title=title, labels={"value": y_title, "variable": "Legend"})
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Times New Roman", size=24, color='black'),
        title_font=dict(size=28, color='black'),
        hoverlabel=dict(bgcolor='white', font_size=20, font_family="Times New Roman", font_color='red'),
        margin=dict(l=60, r=60, t=60, b=60),
    )
    fig.update_traces(line=dict(width=4), hovertemplate='%{y:.2f}')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    st.plotly_chart(fig, use_container_width=True)
    st.download_button(f"📥 Download {filename}", fig.to_image(format="png", engine="kaleido"), file_name=filename, mime="image/png")

# ========== MODE HANDLING ==========

# 1. Prediction and Comparison with Given Actual Value
if mode == "Prediction and Comparison with Given Actual Value":
    file = st.file_uploader("Upload Excel data (with actuals)", type=['xlsx'], key="compare_actual")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = clean_and_round(df)
        forecast_df = forecast_temperature(df)
        comparison_df = pd.merge(df[['Date', 'Te03m', 'Te30m', 'Te50m']], forecast_df, on='Date', how='inner')
        st.write("### 🔍 Predicted vs Actual Sample")
        st.dataframe(comparison_df.head())

# 2. Future Forecasting Only
elif mode == "Future Forecasting Only":
    file = st.file_uploader("Upload temperature file (CSV or Excel)", type=['csv', 'xlsx'], key="forecast")
    if file:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = clean_and_round(df)
        forecast_df = forecast_temperature(df)
        st.write("### 📈 Future Forecast")
        st.dataframe(forecast_df.head())
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast", csv, file_name="forecast.csv", mime="text/csv")

# 3. Compare Predicted with Actual
elif mode == "Compare Predicted with Actual":
    file = st.file_uploader("Upload Excel file with predicted and actual", type=['xlsx'], key="compare")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = clean_and_round(df)
        st.dataframe(df.head())

# 4. Visualize Actual vs Predicted
elif mode == "Visualize Actual vs Predicted":
    file = st.file_uploader("Upload Excel result file", type=['xlsx'], key="vis1")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = clean_and_round(df)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            if f'Actual_{col}' in df.columns and f'Predicted_{col}' in df.columns:
                plot_interactive(df, 'Date', [f'Actual_{col}', f'Predicted_{col}'],
                                 f"Actual vs Predicted for {col} Temperature", "Temperature (°C)",
                                 f"{col}_actual_vs_predicted.png")

# 5. Visualize Accuracy
elif mode == "Visualize Accuracy":
    file = st.file_uploader("Upload accuracy file", type=['xlsx'], key="vis_accuracy")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = clean_and_round(df)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            acc_col = f'Accuracy_{col}'
            if acc_col in df.columns:
                plot_interactive(df, 'Date', [acc_col],
                                 f"Accuracy for {col} Temperature", f"Accuracy_{col} (%)",
                                 f"{col}_accuracy.png")

# 6. Compute SE and MAPE for Each Row
elif mode == "Compute SE and MAPE for Each Row":
    file = st.file_uploader("Upload Excel result file", type=['xlsx'], key="se_mape")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = clean_and_round(df)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            actual = f'Actual_{col}'
            predicted = f'Predicted_{col}'
            if actual in df.columns and predicted in df.columns:
                df[f'Squared_Error_{col}'] = (df[actual] - df[predicted])**2
                df[f'MAPE_{col}'] = np.abs((df[actual] - df[predicted]) / df[actual]) * 100
        st.write("### ✅ Computed SE & MAPE")
        st.dataframe(df.head())

# 7. Calculate Overall Metrics
elif mode == "Calculate Overall Metrics":
    file = st.file_uploader("Upload Excel file with results", type=['xlsx'], key="metrics")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = clean_and_round(df)
        st.write("### 📊 Overall Metrics")
        for col in ['Te03m', 'Te30m', 'Te50m']:
            actual = f'Actual_{col}'
            predicted = f'Predicted_{col}'
            if actual in df.columns and predicted in df.columns:
                rmse = mean_squared_error(df[actual], df[predicted], squared=False)
                mae = mean_absolute_error(df[actual], df[predicted])
                st.write(f"**{col}** — RMSE: {rmse:.2f} °C, MAE: {mae:.2f} °C")

# 8. Visualize Error Metrics
elif mode == "Visualize Error Metrics":
    file = st.file_uploader("Upload Excel file with error metrics", type=['xlsx'], key="error_visual")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = clean_and_round(df)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            for metric in ['Squared_Error', 'MAPE']:
                m_col = f'{metric}_{col}'
                if m_col in df.columns:
                    plot_interactive(df, 'Date', [m_col],
                                     f"{metric.replace('_', ' ')} for {col}", metric, f"{col}_{metric}.png")
