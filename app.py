import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import io
import openpyxl  # Required for reading Excel files

# App title
st.subheader("🛠️ Select Forecasting Mode")

# Mode selection
mode = st.radio(
    "Choose your desired mode:",
    ["Prediction and Comparison with Actual", "Forecasting Only"]
)

# File uploader
uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=["csv", "xlsx"])

# Process uploaded file
if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        except Exception as e:
            st.error("❌ Failed to read Excel file. Please ensure 'openpyxl' is properly installed.")
            st.stop()
    else:
        df = pd.read_csv(uploaded_file)

    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    df.interpolate(method='linear', inplace=True)
    df.bfill(inplace=True)

    # Display uploaded data
    st.subheader("📋 Uploaded Data")
    st.dataframe(df.tail())

    # Model setup
    features = ['Te03m', 'Te30m', 'Te50m']
    look_back = 504
    prediction_horizon = 168
    model = load_model("deep_lstm_checkpoint.keras")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    if mode == "Prediction and Comparison with Actual":
        if len(df) < (look_back + prediction_horizon):
            st.error(f"❌ Need at least {look_back + prediction_horizon} rows for comparison mode.")
        else:
            # Prepare inputs
            X_input = []
            for i in range(len(scaled_data) - look_back):
                X_input.append(scaled_data[i:i+look_back])
            X_input = np.array(X_input)

            # Predict
            prediction = model.predict(X_input)
            prediction = scaler.inverse_transform(prediction)

            # Trim to last 168 for visual
            actual = df[features].iloc[look_back:].values
            dates = df['Date'].iloc[look_back:].reset_index(drop=True)

            prediction = prediction[-prediction_horizon:]
            actual = actual[-prediction_horizon:]
            dates = dates[-prediction_horizon:]

            # Plot prediction vs actual
            st.subheader("📊 Prediction vs Actual")
            fig, ax = plt.subplots(figsize=(15, 6))
            colors = ['red', 'green', 'blue']
            labels = ['Te03m', 'Te30m', 'Te50m']
            for i in range(3):
                ax.plot(dates, actual[:, i], label=f'Actual {labels[i]}', color=colors[i], linewidth=3)
                ax.plot(dates, prediction[:, i], label=f'Predicted {labels[i]}', linestyle='--', color=colors[i], linewidth=3)

            ax.set_title("Forecast vs Actual", fontsize=24, fontweight='bold')
            ax.set_xlabel("Date", fontsize=20)
            ax.set_ylabel("Temperature (°C)", fontsize=20)
            ax.legend(fontsize=14)
            ax.grid(True)
            st.pyplot(fig)

            # Downloadable Excel
            result_df = pd.DataFrame({
                'Date': dates,
                'Actual_Te03m': actual[:, 0],
                'Pred_Te03m': prediction[:, 0],
                'Actual_Te30m': actual[:, 1],
                'Pred_Te30m': prediction[:, 1],
                'Actual_Te50m': actual[:, 2],
                'Pred_Te50m': prediction[:, 2],
            })

            towrite = io.BytesIO()
            result_df.to_excel(towrite, index=False, sheet_name='Comparison')
            towrite.seek(0)
            st.download_button("📥 Download Comparison Results", towrite, file_name="Comparison_Result.xlsx")

    elif mode == "Forecasting Only":
        if len(df) < look_back:
            st.error(f"❌ Need at least {look_back} rows for forecasting.")
        else:
            X_input = scaled_data[-look_back:]
            X_input = np.expand_dims(X_input, axis=0)

            predictions = []
            current_input = X_input.copy()

            for _ in range(prediction_horizon):
                pred = model.predict(current_input)[0][-1]
                predictions.append(pred)
                current_input = np.append(current_input[:, 1:, :], [[pred]], axis=1)

            predictions = scaler.inverse_transform(predictions)
            last_date = df['Date'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=prediction_horizon, freq='H')

            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Pred_Te03m': predictions[:, 0],
                'Pred_Te30m': predictions[:, 1],
                'Pred_Te50m': predictions[:, 2],
            })

            # Plot forecast
            st.subheader("📈 7-Day Forecast")
            fig, ax = plt.subplots(figsize=(15, 6))
            colors = ['red', 'green', 'blue']
            for i, col in enumerate(['Pred_Te03m', 'Pred_Te30m', 'Pred_Te50m']):
                ax.plot(forecast_df['Date'], forecast_df[col], label=col, color=colors[i], linewidth=3)

            ax.set_title("Forecast for Next 168 Steps", fontsize=24, fontweight='bold')
            ax.set_xlabel("Date", fontsize=20)
            ax.set_ylabel("Temperature (°C)", fontsize=20)
            ax.legend(fontsize=14)
            ax.grid(True)
            st.pyplot(fig)

            # Download forecast
            towrite = io.BytesIO()
            forecast_df.to_excel(towrite, index=False, sheet_name='Forecast')
            towrite.seek(0)
            st.download_button("📥 Download Forecast", towrite, file_name="Forecast_168_Steps.xlsx")
