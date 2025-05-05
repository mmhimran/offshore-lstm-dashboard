import streamlit as st
import pandas as pd
import numpy as np
import io
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

# Streamlit Page Config
st.set_page_config(page_title="OFFSHORE TEMPERATURE FORECAST", layout="wide", initial_sidebar_state="expanded")

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deep_lstm_checkpoint.keras")
model = load_model()

# Forecasting
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

# Compare Actual and Predicted
def compare_data(actual_df, predicted_df):
    merged = pd.merge(actual_df, predicted_df, on='Date')
    result = pd.DataFrame({'Date': merged['Date']})
    for col in ['Te03m', 'Te30m', 'Te50m']:
        result[f'Actual_{col}'] = merged[col]
        result[f'Predicted_{col}'] = merged[f'Pred_{col}']
        result[f'Error_{col}'] = result[f'Actual_{col}'] - result[f'Predicted_{col}']
        result[f'Accuracy_{col}'] = 100 - (np.abs(result[f'Error_{col}']) / result[f'Actual_{col}'] * 100)
        result[f'SE_{col}'] = result[f'Error_{col}'] ** 2
        result[f'MAPE_{col}'] = np.abs(result[f'Error_{col}'] / result[f'Actual_{col}']) * 100
    return result

# Round helper
def round_df(df): return df.round(2)

# Excel Export
def generate_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# Plot for Any Metric
def plot_metric(df, y, title):
    fig = go.Figure()

    if isinstance(y, list):
        colors = ['blue', 'red']
        for i, col in enumerate(y):
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df[col], mode='lines',
                name=col, line=dict(color=colors[i], width=4),
                hovertemplate=f'<span style="color:red;"><b>%{{y:.2f}}</b></span><extra></extra>'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df[y], mode='lines',
            name=y, line=dict(color='blue', width=4),
            hovertemplate=f'<span style="color:red;"><b>%{{y:.2f}}</b></span><extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Times New Roman", size=28, color='black'),
            x=0.5
        ),
        xaxis=dict(
            title='Date',
            titlefont=dict(family="Times New Roman", size=24, color='black'),
            tickfont=dict(family="Times New Roman", size=20, color='black'),
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        yaxis=dict(
            title='Temperature (°C)' if 'Temperature' in title else y if isinstance(y, str) else '',
            titlefont=dict(family="Times New Roman", size=24, color='black'),
            tickfont=dict(family="Times New Roman", size=20, color='black'),
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        font=dict(family="Times New Roman", size=20, color='black'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=80, t=80, b=80),
        legend=dict(
            font=dict(family="Times New Roman", size=20, color='black'),
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=20,
            font_family="Times New Roman",
            font_color="red"
        )
    )

    return fig


# UI
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

# 1. Prediction and Comparison with Given Actual
if mode == "Prediction and Comparison with Given Actual Value":
    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if file:
        df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        lookback_df = df.iloc[:504]
        actual_df = df.iloc[504:].reset_index(drop=True)
        forecast_df = forecast_temperature(lookback_df)
        forecast_df.columns = ['Date', 'Pred_Te03m', 'Pred_Te30m', 'Pred_Te50m']
        result = compare_data(actual_df, forecast_df)
        st.download_button("📥 Download Comparison Results", generate_excel(result), file_name="Results_504+168.xlsx")

# 2. Future Forecasting Only
elif mode == "Future Forecasting Only":
    file = st.file_uploader("Upload File", type=["csv", "xlsx"])
    if file:
        df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        forecast = forecast_temperature(df)
        st.download_button("📥 Download Forecast", generate_excel(forecast), file_name="Future_Prediction.xlsx")

# 3. Compare Predicted with Actual
elif mode == "Compare Predicted with Actual":
    actual_file = st.file_uploader("Upload Actual File", type=["csv", "xlsx"], key="actual")
    predicted_file = st.file_uploader("Upload Predicted File", type=["csv", "xlsx"], key="predicted")
    if actual_file and predicted_file:
        actual_df = pd.read_excel(actual_file) if actual_file.name.endswith("xlsx") else pd.read_csv(actual_file)
        predicted_df = pd.read_excel(predicted_file) if predicted_file.name.endswith("xlsx") else pd.read_csv(predicted_file)
        actual_df['Date'] = pd.to_datetime(actual_df['Date'])
        predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])
        predicted_df.columns = ['Date', 'Pred_Te03m', 'Pred_Te30m', 'Pred_Te50m']
        result = compare_data(actual_df, predicted_df)
        st.download_button("📥 Download Result", generate_excel(result), file_name="Result_Comparison.xlsx")

# 4. Visualize Actual vs Predicted
elif mode == "Visualize Actual vs Predicted":
    file = st.file_uploader("Upload Excel result file", type=["xlsx"])
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = round_df(df)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            if f'Actual_{col}' in df.columns and f'Predicted_{col}' in df.columns:
                fig = plot_metric(df, [f'Actual_{col}', f'Predicted_{col}'], f"Actual vs Predicted for {col}")
                st.plotly_chart(fig, use_container_width=True)
                st.download_button(f"📥 Download {col} PNG", fig.to_image(format="png", width=1200, height=600, engine="kaleido"),
                                   file_name=f"{col}_actual_vs_predicted.png", mime="image/png")

# 5. Visualize Accuracy
elif mode == "Visualize Accuracy":
    file = st.file_uploader("Upload Result File", type=["xlsx"])
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = round_df(df)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            if f'Accuracy_{col}' in df.columns:
                fig = plot_metric(df, f'Accuracy_{col}', f"Accuracy for {col}")
                st.plotly_chart(fig, use_container_width=True)

# 6. Compute SE and MAPE for Each Row
elif mode == "Compute SE and MAPE for Each Row":
    file = st.file_uploader("Upload Result File", type=["xlsx"])
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = compare_data(df, df)
        st.download_button("📥 Download Result with SE & MAPE", generate_excel(df), file_name="SE_MAPE_Result.xlsx")

# 7. Calculate Overall Metrics
elif mode == "Calculate Overall Metrics":
    file = st.file_uploader("Upload Result File", type=["xlsx"])
    if file:
        df = pd.read_excel(file)
        for col in ['Te03m', 'Te30m', 'Te50m']:
            mae = mean_absolute_error(df[f'Actual_{col}'], df[f'Predicted_{col}'])
            rmse = np.sqrt(mean_squared_error(df[f'Actual_{col}'], df[f'Predicted_{col}']))
            r2 = 1 - np.sum((df[f'Actual_{col}'] - df[f'Predicted_{col}']) ** 2) / np.sum((df[f'Actual_{col}'] - df[f'Actual_{col}'].mean()) ** 2)
            st.write(f"### 📌 Metrics for {col}")
            st.metric("MAE", f"{mae:.4f}")
            st.metric("RMSE", f"{rmse:.4f}")
            st.metric("R²", f"{r2:.4f}")

# 8. Visualize Error Metrics
elif mode == "Visualize Error Metrics":
    file = st.file_uploader("Upload Result File", type=["xlsx"])
    if file:
        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Te03m', 'Te30m', 'Te50m']:
            for metric in ['SE', 'MAPE']:
                col_name = f"{metric}_{col}"
                if col_name in df.columns:
                    fig = plot_metric(df, col_name, f"{metric} for {col}")
                    st.plotly_chart(fig, use_container_width=True)
