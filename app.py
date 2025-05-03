import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import io

st.set_page_config(layout="wide")
st.subheader("🛠️ Select Forecasting Mode")

mode = st.radio(
    "Choose your desired mode:",
    ["Prediction and Comparison with Actual", "Forecasting Only", "Compare xlsx file"]
)

def calculate_accuracy(y_true, y_pred):
    return 100 - (np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

# Mode 1 & 2: Single file upload
if mode != "Compare xlsx file":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=["csv", "xlsx"])
# Mode 3: Dual file upload
else:
    uploaded_actual = st.file_uploader("Upload Actual file (CSV or Excel)", type=["csv", "xlsx"], key="actual")
    uploaded_pred = st.file_uploader("Upload Predicted file (CSV or Excel)", type=["csv", "xlsx"], key="predicted")

# ------------------------ Mode 1 ------------------------
if uploaded_file and mode == "Prediction and Comparison with Actual":
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df.interpolate(method='linear', inplace=True)
    df.bfill(inplace=True)
    st.subheader("📋 Uploaded Data")
    st.dataframe(df.tail())

    features = ['Te03m', 'Te30m', 'Te50m']
    look_back = 504
    prediction_horizon = 168

    model = load_model("deep_lstm_checkpoint.keras")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    if len(df) < (look_back + prediction_horizon):
        st.error(f"❌ Need at least {look_back + prediction_horizon} rows.")
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
            'Actual_Te30m': actual[:, 1],
            'Pred_Te30m': prediction[:, 1],
            'Actual_Te50m': actual[:, 2],
            'Pred_Te50m': prediction[:, 2],
        })

        for col in ['Te03m', 'Te30m', 'Te50m']:
            result_df[f'Error_{col}'] = result_df[f'Actual_{col}'] - result_df[f'Pred_{col}']
            result_df[f'Accuracy_{col} (%)'] = 100 - (np.abs(result_df[f'Error_{col}']) / result_df[f'Actual_{col}']) * 100

        towrite = io.BytesIO()
        result_df.to_excel(towrite, index=False, sheet_name='Comparison_Result')
        towrite.seek(0)
        st.download_button("📥 Download Prediction vs Actual Results", towrite, file_name="Prediction_Comparison.xlsx")

# ------------------------ Mode 2 ------------------------
elif uploaded_file and mode == "Forecasting Only":
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df.interpolate(method='linear', inplace=True)
    df.bfill(inplace=True)

    features = ['Te03m', 'Te30m', 'Te50m']
    look_back = 504
    prediction_horizon = 168

    model = load_model("deep_lstm_checkpoint.keras")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    if len(df) < look_back:
        st.error(f"❌ Need at least {look_back} rows.")
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
        st.download_button("📥 Download Forecast Only", towrite, file_name="Forecast_168_Steps.xlsx")

# ------------------------ Mode 3 ------------------------
elif mode == "Compare xlsx file" and uploaded_actual and uploaded_pred:
    actual_df = pd.read_excel(uploaded_actual) if uploaded_actual.name.endswith("xlsx") else pd.read_csv(uploaded_actual)
    pred_df = pd.read_excel(uploaded_pred) if uploaded_pred.name.endswith("xlsx") else pd.read_csv(uploaded_pred)

    actual_df['Date'] = pd.to_datetime(actual_df['Date'])
    pred_df['Date'] = pd.to_datetime(pred_df['Date'])

    df = pd.merge(actual_df, pred_df, on="Date", how="inner")
    
    results = pd.DataFrame()
    results['Date'] = df['Date']

    for col in ['Te03m', 'Te30m', 'Te50m']:
        pred_col = f'Pred_{col}'
        error_col = f'Error_{col}'
        acc_col = f'Accuracy_{col} (%)'

        results[f'Actual_{col}'] = df[col]
        results[f'Predicted_{col}'] = df[pred_col]
        results[error_col] = df[col] - df[pred_col]
        results[acc_col] = 100 - (np.abs(results[error_col]) / df[col] * 100)

    results = results.round(4)
    towrite = io.BytesIO()
    results.to_excel(towrite, index=False, sheet_name='Comparison_Result')
    towrite.seek(0)
    st.download_button("📥 Download Comparison Excel", towrite, file_name="Comparison_Result.xlsx")
