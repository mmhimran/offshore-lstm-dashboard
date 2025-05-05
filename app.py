# ============================================================
# OFFSHORE TEMPERATURE FORECAST DASHBOARD (With Plotly & PNG Export)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import tensorflow as tf
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

# --- Set Page Config ---
st.set_page_config(
    page_title="OFFSHORE TEMPERATURE FORECAST",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        .main, .block-container, .css-18e3th9 {
            background-color: #417C7B !important;
        }
        .css-1d391kg, .css-1cpxqw2 {
            color: #FF0000 !important;
            font-size: 18px !important;
        }
        .css-10trblm {
            color: #673AB7 !important;
            font-size: 26px !important;
            font-weight: bold !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FFFAFA !important;
        }
        .css-1x8cf1d {
            color: #673AB7 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load LSTM Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deep_lstm_checkpoint.keras")

model = load_model()

# --- Excel Export Helper ---
def generate_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast')
    return output.getvalue()

# --- Rounding Helper ---
def round_df(df):
    return df.round(2)

# --- Forecast Function ---
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

# --- Compare Actual and Predicted ---
def compare_data(actual_df, predicted_df):
    merged = pd.merge(actual_df, predicted_df, on='Date')
    result = pd.DataFrame({'Date': merged['Date']})
    for col in ['Te03m', 'Te30m', 'Te50m']:
        result[f'Actual_{col}'] = merged[col]
        result[f'Predicted_{col}'] = merged[f'Pred_{col}']
        result[f'Error_{col}'] = result[f'Actual_{col}'] - result[f'Predicted_{col}']
        result[f'Accuracy_{col}'] = 100 - (np.abs(result[f'Error_{col}']) / result[f'Actual_{col}'] * 100)
    return result

# === Title ===
st.title("🧰 OFFSHORE TEMPERATURE FORECAST")

# === Mode Selector ===
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

# === 1. Prediction and Comparison ===
if mode == "Prediction and Comparison with Given Actual Value":
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
        result = compare_data(actual_df, predicted)
        st.download_button("📥 Download Comparison Results", generate_excel(result), file_name="Results_504+168.xlsx")

# === 2. Future Forecasting Only ===
elif mode == "Future Forecasting Only":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=['csv', 'xlsx'])
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        forecast = forecast_temperature(df)
        st.subheader("📋 Uploaded Data")
        st.dataframe(df.tail())
        st.download_button("📥 Download Forecast", generate_excel(forecast), file_name="Result_Prediction_168.xlsx")

# === 3. Compare XLSX ===
elif mode == "Compare Predicted with Actual":
    actual_file = st.file_uploader("Upload Actual File", type=['csv', 'xlsx'], key="actual")
    predicted_file = st.file_uploader("Upload Predicted File", type=['csv', 'xlsx'], key="predicted")
    if actual_file and predicted_file:
        actual_df = pd.read_excel(actual_file) if actual_file.name.endswith('xlsx') else pd.read_csv(actual_file)
        predicted_df = pd.read_excel(predicted_file) if predicted_file.name.endswith('xlsx') else pd.read_csv(predicted_file)
        actual_df['Date'] = pd.to_datetime(actual_df['Date'])
        predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])
        predicted_df.rename(columns=lambda x: f'Pred_{x}' if x != 'Date' else x, inplace=True)
        result = compare_data(actual_df, predicted_df)
        st.download_button("📥 Download Comparison Results", generate_excel(result), file_name="Result_Comparison.xlsx")

# === 4. Visualize Actual vs Predicted ===
elif mode == "Visualize Actual vs Predicted":
    file = st.file_uploader("Upload Excel result file", type=['xlsx'], key="vis1")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = round_df(df)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            fig = px.line(df, x='Date', y=[f'Actual_{col}', f'Predicted_{col}'],
                          labels={'value': 'Temperature (°C)', 'Date': 'Date', 'variable': 'Legend'},
                          title=f"Actual vs Predicted for {col} Temperature")
            fig.update_layout(font=dict(family="Times New Roman", size=20, color='black'),
                              plot_bgcolor='white', paper_bgcolor='white',
                              title_font=dict(size=26, family="Times New Roman", color="black"),
                              hoverlabel=dict(bgcolor="white", font_size=20, font_family="Times New Roman"))
            fig.update_traces(hovertemplate='%{y:.2f}')
            st.plotly_chart(fig, use_container_width=True)
            img_bytes = fig.to_image(format="png", width=1000, height=500, engine="kaleido")
            st.download_button(f"📥 Download {col} Actual vs Predicted Chart as PNG", img_bytes,
                               file_name=f"actual_vs_predicted_{col}.png", mime="image/png")

# === 5. Visualize Accuracy ===
elif mode == "Visualize Accuracy":
    file = st.file_uploader("Upload Excel result file", type=['xlsx'], key="vis2")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = round_df(df)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            fig = px.line(df, x='Date', y=f'Accuracy_{col}', labels={'value': 'Accuracy (%)', 'Date': 'Date'},
                          title=f"Accuracy for {col} Temperature")
            fig.update_layout(font=dict(family="Times New Roman", size=20, color='black'),
                              plot_bgcolor='white', paper_bgcolor='white',
                              title_font=dict(size=26, family="Times New Roman", color="black"),
                              hoverlabel=dict(bgcolor="white", font_size=20, font_family="Times New Roman"))
            fig.update_traces(hovertemplate='%{y:.2f}')
            st.plotly_chart(fig, use_container_width=True)
            img_bytes = fig.to_image(format="png", width=1000, height=500, engine="kaleido")
            st.download_button(f"📥 Download {col} Accuracy Chart as PNG", img_bytes,
                               file_name=f"accuracy_{col}.png", mime="image/png")

# === 6. Compute SE and MAPE for Each Row ===
elif mode == "Compute SE and MAPE for Each Row":
    file = st.file_uploader("Upload Excel result file", type=['xlsx'], key="se_mape")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        result = pd.DataFrame({'Date': df['Date']})
        for col in ['Te03m', 'Te30m', 'Te50m']:
            result[f'Actual_{col}'] = df[f'Actual_{col}']
            result[f'Predicted_{col}'] = df[f'Predicted_{col}']
            result[f'Error_{col}'] = df[f'Actual_{col}'] - df[f'Predicted_{col}']
            result[f'Accuracy_{col}'] = 100 - (np.abs(result[f'Error_{col}']) / df[f'Actual_{col}'] * 100)
            result[f'SE_{col}'] = result[f'Error_{col}'] ** 2
            result[f'MAPE_{col}'] = (np.abs(result[f'Error_{col}']) / df[f'Actual_{col}']) * 100
        st.dataframe(result.head())
        st.download_button("📥 Download Result with SE and MAPE", generate_excel(result), file_name="Result_SE_MAPE.xlsx")

# === 7. Calculate Overall Metrics ===
elif mode == "Calculate Overall Metrics":
    file = st.file_uploader("Upload Excel result file", type=['xlsx'], key="overall_metrics")
    if file:
        df = pd.read_excel(file)
        st.subheader("📊 Overall Evaluation Metrics")
        for col in ['Te03m', 'Te30m', 'Te50m']:
            mae = mean_absolute_error(df[f'Actual_{col}'], df[f'Predicted_{col}'])
            rmse = np.sqrt(mean_squared_error(df[f'Actual_{col}'], df[f'Predicted_{col}']))
            ss_res = np.sum((df[f'Actual_{col}'] - df[f'Predicted_{col}'])**2)
            ss_tot = np.sum((df[f'Actual_{col}'] - df[f'Actual_{col}'].mean())**2)
            r2 = 1 - (ss_res / ss_tot)
            st.write(f"### 📌 {col} Depth")
            st.metric("MAE", f"{mae:.4f}")
            st.metric("RMSE", f"{rmse:.4f}")
            st.metric("R²", f"{r2:.4f}")

# === 8. Visualize Error Metrics ===
elif mode == "Visualize Error Metrics":
    file = st.file_uploader("Upload Excel result file", type=['xlsx'], key="vis_error")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = round_df(df)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            for metric in ['SE', 'MAPE']:
                mcol = f"{metric}_{col}"
                if mcol in df.columns:
                    fig = px.line(df, x='Date', y=mcol, title=f"{metric} for {col}",
                                  labels={'value': metric, 'Date': 'Date'})
                    fig.update_layout(font=dict(family="Times New Roman", size=20, color='black'),
                                      plot_bgcolor='white', paper_bgcolor='white',
                                      title_font=dict(size=26, family="Times New Roman", color="black"),
                                      hoverlabel=dict(bgcolor="white", font_size=20, font_family="Times New Roman"))
                    fig.update_traces(hovertemplate='%{y:.2f}')
                    st.plotly_chart(fig, use_container_width=True)
                    img_bytes = fig.to_image(format="png", width=1000, height=500, engine="kaleido")
                    st.download_button(f"📥 Download {metric} {col} Chart as PNG", img_bytes,
                                       file_name=f"{metric}_{col}.png", mime="image/png")
