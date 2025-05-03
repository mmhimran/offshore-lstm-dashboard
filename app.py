import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from io import BytesIO
import base64

# Custom accuracy calculation
def accuracy(y_true, y_pred):
    return 100 - (np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100)

def generate_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Comparison_Result')
        writer.save()
    processed_data = output.getvalue()
    return processed_data

def download_button(data, filename):
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Comparison Result</a>'
    st.markdown(href, unsafe_allow_html=True)

st.title("🧰 Select Forecasting Mode")

mode = st.radio("Choose your desired mode:", [
    "Prediction and Comparison with Actual",
    "Forecasting Only",
    "Compare xlsx file"
])

if mode == "Prediction and Comparison with Actual":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=['csv', 'xlsx'])
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data")
        st.dataframe(df.tail())

        # Dummy prediction logic for demonstration
        result = df.copy()
        for col in ['Te03m', 'Te30m', 'Te50m']:
            result[f'Pred_{col}'] = df[col].rolling(2).mean().fillna(method='bfill')
            result[f'Error_{col}'] = result[col] - result[f'Pred_{col}']
            result[f'Acc_{col}'] = accuracy(result[col], result[f'Pred_{col}'])

        result = result[[
            'Date',
            'Te03m', 'Pred_Te03m', 'Error_Te03m', 'Acc_Te03m',
            'Te30m', 'Pred_Te30m', 'Error_Te30m', 'Acc_Te30m',
            'Te50m', 'Pred_Te50m', 'Error_Te50m', 'Acc_Te50m'
        ]]

        excel_data = generate_excel(result)
        download_button(excel_data, 'Prediction_vs_Actual.xlsx')

elif mode == "Forecasting Only":
    uploaded_file = st.file_uploader("Upload temperature file (CSV or Excel)", type=['csv', 'xlsx'])
    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data")
        st.dataframe(df.tail())

        # Dummy forecast logic (no actual comparison)
        forecast = df.copy()
        for col in ['Te03m', 'Te30m', 'Te50m']:
            forecast[f'Pred_{col}'] = df[col].rolling(2).mean().fillna(method='bfill')

        forecast = forecast[['Date', 'Pred_Te03m', 'Pred_Te30m', 'Pred_Te50m']]
        excel_data = generate_excel(forecast)
        download_button(excel_data, 'Forecast_Only.xlsx')

elif mode == "Compare xlsx file":
    st.write("Upload Actual file (CSV or Excel)")
    actual_file = st.file_uploader("Upload Actual File", type=['csv', 'xlsx'], key='actual')

    st.write("Upload Predicted file (CSV or Excel)")
    predicted_file = st.file_uploader("Upload Predicted File", type=['csv', 'xlsx'], key='predicted')

    if actual_file and predicted_file:
        actual_df = pd.read_excel(actual_file) if actual_file.name.endswith('.xlsx') else pd.read_csv(actual_file)
        pred_df = pd.read_excel(predicted_file) if predicted_file.name.endswith('.xlsx') else pd.read_csv(predicted_file)

        merged_df = pd.merge(actual_df, pred_df, on='Date', how='inner')
        result = pd.DataFrame({'Date': merged_df['Date']})

        for col in ['Te03m', 'Te30m', 'Te50m']:
            result[f'Actual_{col}'] = merged_df[col]
            result[f'Pred_{col}'] = merged_df[f'Pred_{col}']
            result[f'Error_{col}'] = result[f'Actual_{col}'] - result[f'Pred_{col}']
            result[f'Acc_{col}'] = accuracy(result[f'Actual_{col}'], result[f'Pred_{col}'])

        result = result[[
            'Date',
            'Actual_Te03m', 'Pred_Te03m', 'Error_Te03m', 'Acc_Te03m',
            'Actual_Te30m', 'Pred_Te30m', 'Error_Te30m', 'Acc_Te30m',
            'Actual_Te50m', 'Pred_Te50m', 'Error_Te50m', 'Acc_Te50m'
        ]]

        excel_data = generate_excel(result)
        download_button(excel_data, 'Comparison_Result.xlsx')
