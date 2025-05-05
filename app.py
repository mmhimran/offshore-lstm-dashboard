import streamlit as st
import pandas as pd
import numpy as np
import io
import tensorflow as tf
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

st.set_page_config(
    page_title="OFFSHORE TEMPERATURE FORECAST",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deep_lstm_checkpoint.keras")

model = load_model()

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

def generate_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    return output.getvalue()

def plot_and_download(df, y_cols, title, y_label, filename_prefix):
    fig = px.line(df, x='Date', y=y_cols,
                  labels={'value': y_label, 'Date': 'Date', 'variable': 'Legend'},
                  title=title)
    fig.update_layout(
        font=dict(family="Times New Roman", size=24),
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_font=dict(size=28, family="Times New Roman", color="black"),
        hoverlabel=dict(bgcolor="white", font_size=20, font_family="Times New Roman"),
        legend=dict(font=dict(size=20)),
        margin=dict(t=60)
    )
    fig.update_traces(line=dict(width=4), hovertemplate='%{y:.2f}')
    st.plotly_chart(fig, use_container_width=True)
    img_bytes = fig.to_image(format="png", width=1280, height=720, engine="kaleido")
    st.download_button(f"📥 Download {title} Chart as PNG", img_bytes, file_name=f"{filename_prefix}.png", mime="image/png")

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

if mode == "Prediction and Comparison with Given Actual Value":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=['csv', 'xlsx'])
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        lookback_df = df.iloc[:504]
        actual_df = df.iloc[504:]
        forecast = forecast_temperature(lookback_df)
        predicted = forecast.rename(columns={
            'Te03m': 'Pred_Te03m',
            'Te30m': 'Pred_Te30m',
            'Te50m': 'Pred_Te50m'
        })
        actual_df = actual_df[['Date', 'Te03m', 'Te30m', 'Te50m']].reset_index(drop=True)
        predicted = predicted[['Date', 'Pred_Te03m', 'Pred_Te30m', 'Pred_Te50m']].reset_index(drop=True)
        result = compare_data(actual_df, predicted)
        st.download_button("📥 Download Comparison Results", generate_excel(result), file_name="Results_504+168.xlsx")

elif mode == "Future Forecasting Only":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=['csv', 'xlsx'])
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        forecast = forecast_temperature(df)
        st.download_button("📥 Download Forecast", generate_excel(forecast), file_name="Result_Prediction_168.xlsx")

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

elif mode == "Visualize Actual vs Predicted":
    file = st.file_uploader("Upload Excel result file", type=['xlsx'], key="vis1")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.round(2)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            plot_and_download(df, [f'Actual_{col}', f'Predicted_{col}'], f"Actual vs Predicted for {col} Temperature", "Temperature (°C)", f"actual_vs_predicted_{col}")

elif mode == "Visualize Accuracy":
    file = st.file_uploader("Upload Excel result file", type=['xlsx'], key="vis2")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.round(2)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            plot_and_download(df, [f'Accuracy_{col}'], f"Accuracy for {col} Temperature", "Accuracy (%)", f"accuracy_{col}")

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
            result[f'Accuracy_{col}'] = 100 - (abs(result[f'Error_{col}']) / df[f'Actual_{col}'] * 100)
            result[f'SE_{col}'] = (result[f'Error_{col}'])**2
            result[f'MAPE_{col}'] = abs(result[f'Error_{col}'] / df[f'Actual_{col}']) * 100
        st.download_button("📥 Download Result with SE and MAPE", generate_excel(result), file_name="Result_SE_MAPE.xlsx")

elif mode == "Calculate Overall Metrics":
    file = st.file_uploader("Upload Excel result file", type=['xlsx'], key="overall_metrics")
    if file:
        df = pd.read_excel(file)
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

elif mode == "Visualize Error Metrics":
    file = st.file_uploader("Upload Excel result file", type=['xlsx'], key="vis_error")
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.round(2)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            if f'SE_{col}' in df.columns:
                plot_and_download(df, [f'SE_{col}'], f"Squared Error for {col}", "Squared Error", f"se_{col}")
            if f'MAPE_{col}' in df.columns:
                plot_and_download(df, [f'MAPE_{col}'], f"MAPE for {col}", "MAPE (%)", f"mape_{col}")
