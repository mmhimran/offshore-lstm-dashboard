import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import io
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Page settings
st.set_page_config(page_title="Offshore LSTM Forecast", layout="wide")
st.title("🌊 Offshore Temperature Forecasting Dashboard")
st.markdown("**Deep LSTM-Based Real-Time Forecasting for PETRONAS Flow Assurance**")

# Upload file
uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith("xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # Prepare
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    df.interpolate(method='linear', inplace=True)
    df.bfill(inplace=True)

    st.subheader("🔍 Last 5 Rows of Uploaded Data")
    st.dataframe(df.tail())

    features = ['Te03m', 'Te30m', 'Te50m']
    look_back = 504
    prediction_horizon = 168

    # Scale
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    def create_sequences(data, look_back):
        X = []
        for i in range(len(data) - look_back):
            X.append(data[i:i+look_back])
        return np.array(X)

    X_input = create_sequences(scaled_data, look_back)

    # Load Model
    model = load_model("deep_lstm_checkpoint.keras")

    # Predict
    prediction = model.predict(X_input)
    prediction = scaler.inverse_transform(prediction)

    actual = df[features].iloc[look_back:].values
    dates = df['Date'].iloc[look_back:].reset_index(drop=True)

    # Trim to last 168
    prediction = prediction[-prediction_horizon:]
    actual = actual[-prediction_horizon:]
    dates = dates[-prediction_horizon:]

    # Plot
    st.subheader("📈 LSTM Forecast: Actual vs Predicted (Last 7 Days)")
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ['red', 'green', 'blue']
    depths = ['Te03m', 'Te30m', 'Te50m']
    for i in range(3):
        ax.plot(dates, actual[:, i], label=f'Actual {depths[i]}', color=colors[i], linewidth=3)
        ax.plot(dates, prediction[:, i], label=f'Predicted {depths[i]}', color=colors[i], linestyle='--', linewidth=3)

    ax.set_title("Offshore Forecast: Actual vs Predicted", fontsize=24, fontweight='bold', fontname='Times New Roman')
    ax.set_xlabel("Date", fontsize=20, fontweight='bold', fontname='Times New Roman')
    ax.set_ylabel("Temperature (°C)", fontsize=20, fontweight='bold', fontname='Times New Roman')
    ax.legend(fontsize=14)
    ax.grid(True)
    st.pyplot(fig)

    # Errors
    rmse = np.sqrt(mean_squared_error(actual, prediction))
    mae = mean_absolute_error(actual, prediction)
    st.success(f"✅ RMSE: {rmse:.4f} °C | MAE: {mae:.4f} °C")

    # Save Excel
    result_df = pd.DataFrame({
        'Date': dates,
        'Actual_Te03m': actual[:, 0],
        'Predicted_Te03m': prediction[:, 0],
        'Error_Te03m': actual[:, 0] - prediction[:, 0],
        'Actual_Te30m': actual[:, 1],
        'Predicted_Te30m': prediction[:, 1],
        'Error_Te30m': actual[:, 1] - prediction[:, 1],
        'Actual_Te50m': actual[:, 2],
        'Predicted_Te50m': prediction[:, 2],
        'Error_Te50m': actual[:, 2] - prediction[:, 2],
    })

    towrite = io.BytesIO()
    result_df.to_excel(towrite, index=False, sheet_name='Forecast Results')
    towrite.seek(0)

    st.download_button("📥 Download Excel Results", towrite, file_name="LSTM_Offshore_Forecast.xlsx")
