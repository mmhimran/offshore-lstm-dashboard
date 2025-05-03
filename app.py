import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Title
st.subheader("\U0001F6E0\ufe0f Select Forecasting Mode")

# Mode selection
mode = st.radio("Choose your desired mode:", [
    "Prediction and Comparison with Actual",
    "Forecasting Only",
    "Comparison Only"
])

# File uploader
uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Read the file
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        df = pd.read_csv(uploaded_file)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    if mode == "Prediction and Comparison with Actual":
        st.subheader("\U0001F4C4 Uploaded Data")
        st.dataframe(df.tail())

        features = ['Te03m', 'Te30m', 'Te50m']
        look_back = 504
        prediction_horizon = 168

        model = load_model("deep_lstm_checkpoint.keras")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[features])

        if len(df) < (look_back + prediction_horizon):
            st.error(f"❌ Need at least {look_back + prediction_horizon} rows for comparison mode.")
        else:
            X_input = []
            for i in range(len(scaled_data) - look_back):
                X_input.append(scaled_data[i:i+look_back])
            X_input = np.array(X_input)

            prediction = model.predict(X_input)
            prediction = scaler.inverse_transform(prediction)

            actual = df[features].iloc[look_back:].values
            dates = df['Date'].iloc[look_back:].reset_index(drop=True)

            prediction = prediction[-prediction_horizon:]
            actual = actual[-prediction_horizon:]
            dates = dates[-prediction_horizon:]

            result_df = pd.DataFrame({
                'Date': dates,
                'Actual_Te03m': actual[:, 0],
                'Pred_Te03m': prediction[:, 0],
                'Error_Te03m': actual[:, 0] - prediction[:, 0],
                'Accuracy_Te03m (%)': 100 - np.abs((actual[:, 0] - prediction[:, 0]) / actual[:, 0]) * 100,
                'Actual_Te30m': actual[:, 1],
                'Pred_Te30m': prediction[:, 1],
                'Error_Te30m': actual[:, 1] - prediction[:, 1],
                'Accuracy_Te30m (%)': 100 - np.abs((actual[:, 1] - prediction[:, 1]) / actual[:, 1]) * 100,
                'Actual_Te50m': actual[:, 2],
                'Pred_Te50m': prediction[:, 2],
                'Error_Te50m': actual[:, 2] - prediction[:, 2],
                'Accuracy_Te50m (%)': 100 - np.abs((actual[:, 2] - prediction[:, 2]) / actual[:, 2]) * 100
            })

            # Downloadable Excel
            towrite = io.BytesIO()
            result_df.to_excel(towrite, index=False, sheet_name='Comparison')
            towrite.seek(0)
            st.download_button("📥 Download Comparison Results", towrite, file_name="Comparison_Result.xlsx")

    elif mode == "Forecasting Only":
        st.subheader("\U0001F4C4 Uploaded Data")
        st.dataframe(df.tail())

        features = ['Te03m', 'Te30m', 'Te50m']
        look_back = 504
        prediction_horizon = 168

        model = load_model("deep_lstm_checkpoint.keras")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[features])

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

            towrite = io.BytesIO()
            forecast_df.to_excel(towrite, index=False, sheet_name='Forecast')
            towrite.seek(0)
            st.download_button("📥 Download Forecast", towrite, file_name="Forecast_168_Steps.xlsx")

    elif mode == "Comparison Only":
        st.subheader("\U0001F4C4 Uploaded Data")
        st.dataframe(df.tail())

        # Expect columns: Date, Actual_Te03m, Pred_Te03m, ...
        features = ['Te03m', 'Te30m', 'Te50m']

        for f in features:
            df[f'Error_{f}'] = df[f'Actual_{f}'] - df[f'Pred_{f}']
            df[f'Accuracy_{f} (%)'] = 100 - np.abs((df[f'Error_{f}']) / df[f'Actual_{f}']) * 100

        # Reorder columns
        ordered_cols = ['Date']
        for f in features:
            ordered_cols += [f'Actual_{f}', f'Pred_{f}', f'Error_{f}', f'Accuracy_{f} (%)']
        df = df[ordered_cols]

        towrite = io.BytesIO()
        df.to_excel(towrite, index=False, sheet_name='ComparisonOnly')
        towrite.seek(0)
        st.download_button("📥 Download Comparison Report", towrite, file_name="Comparison_Only.xlsx")
