# OFFSHORE TEMPERATURE FORECAST DASHBOARD

import streamlit as st
import pandas as pd
import numpy as np
import io
import tensorflow as tf
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

# --- Page config ---
st.set_page_config(
    page_title="OFFSHORE TEMPERATURE FORECAST",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS style ---
st.markdown("""
    <style>
        .main, .block-container, .css-18e3th9 {
            background-color: #417C7B !important;
        }
        .css-1d391kg, .css-1cpxqw2 {
            color: #FFFFFF !important;
            font-size: 18px !important;
        }
        .css-10trblm {
            color: #FFFAFA !important;
            font-size: 26px !important;
            font-weight: bold !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FFFAFA !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deep_lstm_checkpoint.keras")

model = load_model()

# --- Helpers ---
def generate_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def round_df(df):
    return df.round(2)

def clean_and_round(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return round_df(df)

def show_preview(df):
    st.subheader("📋 Preview of Uploaded Data")
    st.dataframe(df.head())

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

def compare_data(actual_df, predicted_df):
    merged = pd.merge(actual_df, predicted_df, on='Date')
    result = pd.DataFrame({'Date': merged['Date']})
    for col in ['Te03m', 'Te30m', 'Te50m']:
        result[f'Actual_{col}'] = merged[col]
        result[f'Predicted_{col}'] = merged[f'Pred_{col}']
        result[f'Error_{col}'] = result[f'Actual_{col}'] - result[f'Predicted_{col}']
        result[f'Accuracy_{col}'] = 100 - (np.abs(result[f'Error_{col}']) / result[f'Actual_{col}'] * 100)
    return result

# --- UI Title ---
st.title("🧰 OFFSHORE TEMPERATURE FORECAST")

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
        df = clean_and_round(df)
        show_preview(df)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            if f'Actual_{col}' in df.columns and f'Predicted_{col}' in df.columns:
                fig = px.line(df, x='Date', y=[f'Actual_{col}', f'Predicted_{col}'],
                              title=f"Actual vs Predicted for {col} Temperature",
                              labels={'value': 'Temperature (°C)', 'Date': 'Date'})
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
        df = clean_and_round(df)
        show_preview(df)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            acc_col = f'Accuracy_{col}'
            if acc_col in df.columns:
                fig = px.line(df, x='Date', y=acc_col,
                              title=f"Accuracy for {col} Temperature",
                              labels={'value': 'Accuracy (%)', 'Date': 'Date'})
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
        result = pd.DataFrame()
        result['Date'] = df['Date']
        for col in ['Te03m', 'Te30m', 'Te50m']:
            actual_col = f'Actual_{col}'
            pred_col = f'Predicted_{col}'
            error_col = f'Error_{col}'
            acc_col = f'Accuracy_{col}'
            se_col = f'SE_{col}'
            mape_col = f'MAPE_{col}'
            result[actual_col] = df[actual_col]
            result[pred_col] = df[pred_col]
            result[error_col] = df[actual_col] - df[pred_col]
            result[acc_col] = 100 - (abs(result[error_col]) / df[actual_col] * 100)
            result[se_col] = (result[error_col]) ** 2
            result[mape_col] = abs(result[error_col] / df[actual_col]) * 100
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
        df = clean_and_round(df)
        show_preview(df)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            se_col = f'SE_{col}'
            mape_col = f'MAPE_{col}'
            if se_col in df.columns:
                fig_se = px.line(df, x='Date', y=se_col,
                                 title=f"Squared Error for {col}",
                                 labels={'value': 'Squared Error', 'Date': 'Date'})
                fig_se.update_traces(hovertemplate='%{y:.2f}')
                st.plotly_chart(fig_se, use_container_width=True)
                img_bytes = fig_se.to_image(format="png", width=1000, height=500, engine="kaleido")
                st.download_button(f"📥 Download {col} SE Chart as PNG", img_bytes, file_name=f"se_{col}.png")

            if mape_col in df.columns:
                fig_mape = px.line(df, x='Date', y=mape_col,
                                   title=f"MAPE for {col}",
                                   labels={'value': 'MAPE (%)', 'Date': 'Date'})
                fig_mape.update_traces(hovertemplate='%{y:.2f}')
                st.plotly_chart(fig_mape, use_container_width=True)
                img_bytes = fig_mape.to_image(format="png", width=1000, height=500, engine="kaleido")
                st.download_button(f"📥 Download {col} MAPE Chart as PNG", img_bytes, file_name=f"mape_{col}.png")
